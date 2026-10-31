//! Bridge code generation from parsed metadata.
//!
//! `generate_batch_impl` takes batched metadata token streams and produces
//! the corresponding `extern "C"` FFI functions.

use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, format_ident, quote};

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::meta::{
    HasPrefix, MetaBitflags, MetaEnum, MetaError, MetaExportable, MetaFreeFunction,
    MetaImplementable, MetaMethod, MetaMethodContext, MetaParam, MetaParamKind, MetaReceiver,
    MetaReturn, MetaTraitImpl, MetaTypePair, camel_to_snake, camel_to_upper_snake,
    cfg_predicate_attr, extract_result_ok_type, is_result_ok_handle, peek_meta_field,
    peek_meta_tag,
};

/// Maps trait names to their concrete dispatch variants.
type TraitMap = HashMap<String, TraitDispatchInfo>;

/// Maps error type names to their metadata (type tag, path, variants).
///
/// Used by bridge generation to emit `ffier_result(type_tag, code)`
/// with the correct type tag for `Result<T, E>` returns.
type ErrorMap = HashMap<String, ErrorInfo>;

struct ErrorInfo {
    pub type_tag: u32,
}

struct TraitDispatchInfo {
    pub variants: Vec<TraitVariant>,
    /// If the trait has an `@implementable` metadata entry, trait-level dispatch info.
    pub implementable: Option<ImplementableInfo>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TraitVariantKind {
    Concrete,
    Wrapper,
}

struct TraitVariant {
    pub name: String,
    pub bridge_type: TokenStream2,
    pub kind: TraitVariantKind,
    /// `#[cfg(...)]` predicate. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream2>,
}

struct ImplementableInfo {
    pub trait_path: TokenStream2,

    pub methods: Vec<MetaMethod>,
    /// Number of methods that belong to this trait (not supertrait methods).
    /// Only the first `own_method_count` methods are dispatched in self-dispatch
    /// functions. Supertrait methods need separate dispatch through their own trait.
    pub own_method_count: usize,
    /// `#[cfg(...)]` predicate for the entire trait. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream2>,
}

/// Build the trait-to-impls map from parsed `@implementable` and `@trait_impl` metadata.
fn build_trait_map(implementables: &[TokenStream2], trait_impls: &[TokenStream2]) -> TraitMap {
    let mut map = TraitMap::new();

    // trait_impl entries: "Fruit for Apple" → Apple is a concrete implementor
    for item in trait_impls {
        if let Ok(meta) = syn::parse2::<MetaTraitImpl>(item.clone()) {
            let trait_name = meta.trait_name.to_string();
            let struct_name = meta.struct_name.to_string();
            let struct_path = meta.struct_path;
            let cfg_predicate = meta.cfg_predicate;
            map.entry(trait_name)
                .or_insert_with(|| TraitDispatchInfo {
                    variants: Vec::new(),
                    implementable: None,
                })
                .variants
                .push(TraitVariant {
                    name: struct_name,
                    bridge_type: struct_path,
                    kind: TraitVariantKind::Concrete,
                    cfg_predicate,
                });
        }
    }

    // implementable entries: "trait Fruit" → adds VtableFruit wrapper + stores vtable info
    for item in implementables {
        if let Ok(meta) = syn::parse2::<MetaImplementable>(item.clone()) {
            let trait_name = meta.trait_name.to_string();
            let wrapper_name = format!("Vtable{trait_name}");
            let wrapper_path = meta.wrapper_name.clone();
            let methods = meta.methods;
            let own_method_count = meta.own_method_count;
            let no_vtable = meta.no_vtable;

            let info = map.entry(trait_name).or_insert_with(|| TraitDispatchInfo {
                variants: Vec::new(),
                implementable: None,
            });
            // no_vtable traits have no VtableWrapper type — C callers can't
            // implement them, only concrete Rust types are dispatched.
            if !no_vtable {
                info.variants.push(TraitVariant {
                    name: wrapper_name,
                    bridge_type: wrapper_path.clone(),
                    kind: TraitVariantKind::Wrapper,
                    cfg_predicate: meta.cfg_predicate.clone(),
                });
            }
            info.implementable = Some(ImplementableInfo {
                trait_path: meta.trait_path,
                methods,
                own_method_count,
                cfg_predicate: meta.cfg_predicate,
            });
        }
    }

    map
}

/// Wrap a multi-item token stream in `#[cfg(PRED)] const _: () = { ... };`
/// so all items are gated as a unit. Returns `code` unchanged if `cfg_attr`
/// is empty (no cfg predicate).
fn wrap_cfg(cfg_attr: Option<TokenStream2>, code: TokenStream2) -> TokenStream2 {
    match cfg_attr {
        Some(attr) => quote! { #attr const _: () = { #code }; },
        None => code,
    }
}

fn generate_one(
    item: TokenStream2,
    trait_map: &TraitMap,
    error_map: &ErrorMap,
    handle_types: &HashSet<String>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let tag = peek_meta_tag(&item);
    match tag.as_str() {
        "exported_impl" => {
            let meta: MetaExportable = match syn::parse2(item) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let cfg_attr = cfg_predicate_attr(meta.cfg_predicate.as_ref());
            let code =
                generate_exportable_bridge(meta, trait_map, error_map, handle_types, lib_crate);
            wrap_cfg(cfg_attr, code)
        }
        "exported_error" => {
            let meta: MetaError = match syn::parse2(item) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            generate_error_bridge(meta, lib_crate)
        }
        "exported_trait" => {
            let meta: MetaImplementable = match syn::parse2(item) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let cfg_attr = cfg_predicate_attr(meta.cfg_predicate.as_ref());
            let code = generate_implementable_bridge(meta, lib_crate);
            wrap_cfg(cfg_attr, code)
        }
        "exported_trait_impl" => {
            let meta: MetaTraitImpl = match syn::parse2(item) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let cfg_attr = cfg_predicate_attr(meta.cfg_predicate.as_ref());
            let code =
                generate_trait_impl_bridge(meta, trait_map, error_map, handle_types, lib_crate);
            wrap_cfg(cfg_attr, code)
        }
        "exported_enum" | "exported_bitflags" => {
            // No bridge code needed — enums/bitflags are value types passed by value.
            quote! {}
        }
        "exported_fn" => {
            let meta: MetaFreeFunction = match syn::parse2(item) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let cfg_attr = cfg_predicate_attr(meta.cfg_predicate.as_ref());
            let code = generate_free_fn_bridge(meta, error_map, handle_types, trait_map, lib_crate);
            wrap_cfg(cfg_attr, code)
        }
        _ => {
            let msg = format!("unknown metadata tag `@{tag}`");
            quote! { compile_error!(#msg); }
        }
    }
}

/// Extract a string literal value from a token tree, unwrapping transparent
/// (None-delimited) groups that `macro_rules!` wraps around captured fragments.
/// Uses `syn::LitStr` for proper unescaping.
fn unwrap_literal(tt: Option<proc_macro2::TokenTree>) -> Option<String> {
    match tt {
        Some(proc_macro2::TokenTree::Literal(lit)) => {
            let stream = proc_macro2::TokenStream::from(proc_macro2::TokenTree::Literal(lit));
            syn::parse2::<syn::LitStr>(stream).ok().map(|s| s.value())
        }
        Some(proc_macro2::TokenTree::Group(g)) if g.delimiter() == proc_macro2::Delimiter::None => {
            // Recurse into the transparent group
            unwrap_literal(g.stream().into_iter().next())
        }
        _ => None,
    }
}

/// Generates bridge code from batched metadata items.
///
/// Input: `{ @tag, ... } { @tag, ... } ...` — multiple brace-delimited items.
/// Sorts into errors → exportables → implementables → trait_impls, generates
/// bridge code for each.
pub fn generate_batch_impl(input: TokenStream2) -> TokenStream2 {
    // Parse @lib_crate = path; @primitives_prefix = "..."; from chain macro
    let mut iter = input.into_iter().peekable();
    let lib_crate: TokenStream2 = {
        let mut path_tokens = Vec::new();
        let mut found = false;
        if let Some(proc_macro2::TokenTree::Punct(p)) = iter.peek()
            && p.as_char() == '@'
        {
            found = true;
            iter.next(); // @
            iter.next(); // lib_crate
            iter.next(); // =
            for tt in iter.by_ref() {
                if let proc_macro2::TokenTree::Punct(p) = &tt
                    && p.as_char() == ';'
                {
                    break;
                }
                path_tokens.push(tt);
            }
        }
        if !found {
            return quote! { compile_error!("missing @lib_crate prefix in bridge metadata"); };
        }
        path_tokens.into_iter().collect()
    };
    // Parse optional @-prefixed preamble keys.
    // Known keys: @primitives_prefix, @schema_output.
    let mut primitives_prefix: Option<String> = None;
    let mut schema_output: Option<String> = None;
    while let Some(proc_macro2::TokenTree::Punct(p)) = iter.peek()
        && p.as_char() == '@'
    {
        iter.next(); // @
        let key = match iter.next() {
            Some(proc_macro2::TokenTree::Ident(id)) => id.to_string(),
            _ => {
                return quote! { compile_error!("expected identifier after `@` in preamble"); };
            }
        };
        match iter.next() {
            Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {}
            _ => {
                let msg = format!("expected `=` after @{key}");
                return quote! { compile_error!(#msg); };
            }
        }
        let value = match unwrap_literal(iter.next()) {
            Some(s) => s,
            None => {
                let msg = format!("expected string literal for @{key}");
                return quote! { compile_error!(#msg); };
            }
        };
        match iter.next() {
            Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ';' => {}
            _ => {
                let msg = format!("expected `;` after @{key} value");
                return quote! { compile_error!(#msg); };
            }
        }
        match key.as_str() {
            "primitives_prefix" => primitives_prefix = Some(value),
            "schema_output" => schema_output = Some(value),
            _ => {
                let msg = format!("unknown preamble key @{key}");
                return quote! { compile_error!(#msg); };
            }
        }
    }
    let schema_output = match schema_output {
        Some(s) => s,
        None => return quote! { compile_error!("missing @schema_output in bridge metadata"); },
    };

    let mut items: Vec<TokenStream2> = Vec::new();
    for tt in iter {
        if let proc_macro2::TokenTree::Group(g) = tt
            && g.delimiter() == proc_macro2::Delimiter::Brace
        {
            items.push(g.stream());
        }
    }

    // Sort by category for correct ordering
    let mut errors = Vec::new();
    let mut exportables = Vec::new();
    let mut implementables = Vec::new();
    let mut trait_impls = Vec::new();
    let mut enum_constants = Vec::new();
    let mut bitflags_constants = Vec::new();
    let mut free_fns = Vec::new();

    for item in &items {
        match peek_meta_tag(item).as_str() {
            "exported_error" => errors.push(item.clone()),
            "exported_impl" => exportables.push(item.clone()),
            "exported_trait" => implementables.push(item.clone()),
            "exported_trait_impl" => trait_impls.push(item.clone()),
            "exported_enum" => enum_constants.push(item.clone()),
            "exported_bitflags" => bitflags_constants.push(item.clone()),
            "exported_fn" => free_fns.push(item.clone()),
            tag => {
                let msg = format!("unknown metadata tag `@{tag}` in batch");
                return quote! { compile_error!(#msg); };
            }
        }
    }

    // Pass 1: Build trait-to-impls map from @trait_impl and @implementable entries.
    // This allows resolving `impl Trait` params automatically.
    let trait_map = build_trait_map(&implementables, &trait_impls);

    // Build error map: error name → (type_tag, path)
    let error_map: ErrorMap = {
        let mut map = ErrorMap::new();
        for item in &errors {
            if let Ok(meta) = syn::parse2::<MetaError>(item.clone()) {
                map.insert(
                    meta.name.to_string(),
                    ErrorInfo {
                        type_tag: meta.type_tag,
                    },
                );
            }
        }
        map
    };

    // Build handle set: type names that are opaque handles (exportables + implementables).
    // Used to determine handle-or-null returns for Result<Handle, E>.
    let handle_types: HashSet<String> = {
        let mut set = HashSet::new();
        for item in &exportables {
            if let Ok(meta) = syn::parse2::<MetaExportable>(item.clone()) {
                set.insert(meta.struct_name.to_string());
            }
        }
        for item in &implementables {
            if let Ok(meta) = syn::parse2::<MetaImplementable>(item.clone()) {
                // Vtable wrapper types are also handles
                set.insert(format!("Vtable{}", meta.trait_name));
            }
        }
        set
    };

    // Pass 1.5: Validate type tags — check for missing (tag=0) and duplicates.
    // Also builds a tag→name map used to generate __ffier_type_name() for
    // human-readable panic messages on type mismatch.
    let mut tag_to_name: HashMap<u32, String> = HashMap::new();
    {
        let mut check_tag = |tag: u32, name: &str, hint: &str| -> Option<TokenStream2> {
            if tag == 0 {
                let msg =
                    format!("type `{name}` has no type tag; add `{hint}` in library_definition!()");
                return Some(quote! { compile_error!(#msg); });
            }
            if let Some(prev) = tag_to_name.get(&tag) {
                let msg = format!("duplicate type tag {tag}: used by both `{prev}` and `{name}`");
                return Some(quote! { compile_error!(#msg); });
            }
            tag_to_name.insert(tag, name.to_string());
            None
        };

        for item in &errors {
            let meta: MetaError = match syn::parse2(item.clone()) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let name = meta.name.to_string();
            if let Some(err) = check_tag(meta.type_tag, &name, &format!("{name} = N")) {
                return err;
            }
        }

        for item in &exportables {
            let meta: MetaExportable = match syn::parse2(item.clone()) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let name = meta.struct_name.to_string();
            if let Some(err) = check_tag(meta.type_tag, &name, &format!("{name} = N")) {
                return err;
            }
        }

        for item in &implementables {
            let meta: MetaImplementable = match syn::parse2(item.clone()) {
                Ok(m) => m,
                Err(e) => return e.to_compile_error(),
            };
            let name = format!("Vtable{}", meta.trait_name);
            if let Some(err) = check_tag(
                meta.type_tag,
                &name,
                &format!("trait {} = N", meta.trait_name),
            ) {
                return err;
            }
        }
    }

    // Generate __ffier_type_name(), __ffier_dispatch_panic(), and per-trait
    // accepted-types constants. These are used by all dispatch error messages
    // (self-dispatch, impl Trait, type assertions) for consistent, human-readable
    // panic messages.
    let dispatch_helpers = {
        let mut sorted: Vec<_> = tag_to_name.iter().collect();
        sorted.sort_by_key(|(tag, _)| *tag);
        let tags: Vec<u32> = sorted.iter().map(|(t, _)| **t).collect();
        let names: Vec<&str> = sorted.iter().map(|(_, n)| n.as_str()).collect();

        // Generate one const per trait: __FFIER_ACCEPTED_Fruit = "Apple | Orange | VtableFruit"
        let accepted_consts: Vec<TokenStream2> = trait_map
            .iter()
            .map(|(trait_name, info)| {
                let const_name = format_ident!("__FFIER_ACCEPTED_{trait_name}");
                let accepted = info
                    .variants
                    .iter()
                    .map(|v| v.name.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ");
                quote! {
                    const #const_name: &str = #accepted;
                }
            })
            .collect();

        quote! {
            /// Look up the type name for a given type tag. Returns "unknown" for
            /// unrecognized tags (e.g. corrupted handle or use-after-free).
            fn __ffier_type_name(tag: u32) -> &'static str {
                match tag {
                    #(#tags => #names,)*
                    _ => "unknown",
                }
            }

            /// Panic with a clear message when a handle's type tag doesn't match
            /// any expected type.
            ///
            /// - `fn_name`: the C function name (e.g. "ft_fruit_value")
            /// - `expected`: what was expected (e.g. "Fruit implementor")
            /// - `accepted`: list of accepted type names (e.g. "Apple | Orange | VtableFruit"),
            ///   or empty string if not applicable (e.g. for single-type assertions)
            /// - `actual_tag`: the type tag read from the handle
            #[cold]
            #[inline(never)]
            fn __ffier_dispatch_panic(
                fn_name: &str,
                expected: &str,
                accepted: &str,
                actual_tag: u32,
            ) -> ! {
                let actual_name = __ffier_type_name(actual_tag);
                let actual_lib = ffier::extract_library_tag(actual_tag);
                // Detect cross-library handle confusion: if the library_tag
                // doesn't match any known type in this library, the handle
                // likely belongs to a different ffier library.
                let cross_lib_hint = if actual_name == "unknown" && actual_lib != 0 {
                    " (handle appears to belong to a different ffier library)"
                } else {
                    ""
                };
                if accepted.is_empty() {
                    panic!(
                        "{}(): expected {}, got {} (type_tag=0x{:08x}){}",
                        fn_name, expected, actual_name, actual_tag, cross_lib_hint,
                    );
                } else {
                    panic!(
                        "{}(): expected {} ({}), got {} (type_tag=0x{:08x}){}",
                        fn_name, expected, accepted, actual_name, actual_tag, cross_lib_hint,
                    );
                }
            }

            #(#accepted_consts)*
        }
    };

    // Pass 2: Generate bridge code for each item in sorted order
    let mut all_code = Vec::new();

    for item in errors
        .iter()
        .chain(exportables.iter())
        .chain(implementables.iter())
        .chain(trait_impls.iter())
        .chain(enum_constants.iter())
        .chain(bitflags_constants.iter())
        .chain(free_fns.iter())
    {
        all_code.push(generate_one(
            item.clone(),
            &trait_map,
            &error_map,
            &handle_types,
            &lib_crate,
        ));
    }

    // Extract prefix from any item for shared types (all items share the same prefix)
    let first_prefix = errors
        .iter()
        .chain(exportables.iter())
        .chain(implementables.iter())
        .chain(trait_impls.iter())
        .chain(enum_constants.iter())
        .chain(bitflags_constants.iter())
        .chain(free_fns.iter())
        .next()
        .map(|item| peek_meta_field(item, "prefix"))
        .unwrap_or_default();

    // Debug utility: exported function to inspect a handle's type tag and name.
    let debug_fn = {
        let debug_fn_name = format_ident!("{first_prefix}_debug_handle_type");
        quote! {
            /// Debug utility: read the type tag from a handle and return the
            /// type name as a string. Returns "null" for null handles,
            /// "unknown" for unrecognized tags.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #debug_fn_name(
                handle: *const core::ffi::c_void,
            ) -> ffier::FfierBytes {
                if handle.is_null() {
                    return unsafe { ffier::FfierBytes::from_str("null") };
                }
                let tag = unsafe { ffier::handle_type_tag(handle) };
                let name = __ffier_type_name(tag);
                unsafe { ffier::FfierBytes::from_str(name) }
            }
        }
    };

    // Pass 3: Generate self-dispatch functions for exported traits.
    // For each trait with an @implementable entry, generate per-trait dispatching
    // C functions that read the type tag and dispatch to the concrete implementor.
    for (trait_name, info) in &trait_map {
        if let Some(impl_info) = &info.implementable {
            let cfg_attr = cfg_predicate_attr(impl_info.cfg_predicate.as_ref());
            let code = generate_self_dispatch_bridge(
                trait_name,
                info,
                &first_prefix,
                &trait_map,
                &error_map,
                &handle_types,
                &lib_crate,
            );
            all_code.push(wrap_cfg(cfg_attr, code));
        }
    }

    // Generate strerror dispatch function — dispatches by type tag to per-error
    // static_message tables.
    let strerror_fn = generate_strerror_bridge(&first_prefix, &errors, &lib_crate);

    // Generate per-variant field getters for data-carrying error variants
    let error_getters = generate_error_getters(&first_prefix, &errors, &lib_crate);

    // Generate str_free function for dropping owned strings (Box<str>)
    let str_free_fn = generate_str_free(&first_prefix);

    // Generate free_object_array function
    let free_object_array_fn = generate_free_object_array(&first_prefix);

    // Emit JSON schema to the user-specified path.
    emit_json(
        &first_prefix,
        primitives_prefix.as_deref(),
        &schema_output,
        &errors,
        &exportables,
        &implementables,
        &trait_impls,
        &enum_constants,
        &bitflags_constants,
        &free_fns,
    );

    quote! {
        #dispatch_helpers

        #debug_fn

        #(#all_code)*

        #strerror_fn

        #error_getters

        #str_free_fn

        #free_object_array_fn
    }
}

// ===========================================================================
// Error payload getter — borrows field data from the handle
// ===========================================================================

/// Generate `{prefix}_error_payload(handle, out_buf, buf_size)` — shallow-
/// copies the variant's CRepr into caller-provided storage, *borrowing* the
/// data from the handle (handle stays alive, caller must not outlive it).
///
/// For `Box<str>` payloads this writes a `FfierBytes { data, len }` pointing
/// into the handle's owned string. Fieldless variants are a no-op.
fn generate_error_getters(
    prefix: &str,
    errors: &[TokenStream2],
    _lib_crate: &TokenStream2,
) -> TokenStream2 {
    let fn_pfx = format!("{prefix}_");
    let payload_fn = format_ident!("{fn_pfx}error_payload");

    let mut dispatch_arms = Vec::new();

    for item in errors {
        let Ok(meta) = syn::parse2::<MetaError>(item.clone()) else {
            continue;
        };
        let type_tag = meta.type_tag;
        let path = &meta.path;

        dispatch_arms.push(quote! {
            #type_tag => {
                let err: &#path = unsafe { ffier::ffier_handle_borrow(handle) };
                unsafe { ffier::FfiError::payload(err, out_buf, buf_size) };
            }
        });
    }

    quote! {
        /// Shallow-copy the error variant's payload into caller-provided
        /// storage. The written CRepr borrows from the handle — the
        /// handle must stay alive while the caller uses the data.
        ///
        /// Fieldless variants are a no-op (out_buf untouched).
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #payload_fn(
            handle: *const core::ffi::c_void,
            out_buf: *mut core::ffi::c_void,
            buf_size: usize,
        ) {
            if handle.is_null() { return; }
            let type_tag = unsafe { ffier::handle_type_tag(handle) };
            match type_tag {
                #(#dispatch_arms)*
                _ => {}
            }
        }
    }
}

// ===========================================================================
// str_free — drop an owned string (Box<str>) returned across FFI
// ===========================================================================

/// Generate `{prefix}_str_free(FfierBytes s)` that reconstitutes and drops a
/// `Box<str>` previously leaked via `Box::leak` in a `Box<str>` return.
fn generate_str_free(prefix: &str) -> TokenStream2 {
    let fn_pfx = format!("{prefix}_");
    let str_free_fn = format_ident!("{fn_pfx}str_free");

    quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #str_free_fn(s: ffier::FfierBytes) {
            if !s.data.is_null() {
                unsafe {
                    let slice = core::slice::from_raw_parts_mut(s.data as *mut u8, s.len);
                    drop(Box::from_raw(core::str::from_utf8_unchecked_mut(slice)));
                }
            }
        }
    }
}

// ===========================================================================
// free_object_array — free a FfierObjectArray
// ===========================================================================

/// Generate `{prefix}_free_object_array(FfierObjectArray a)` that frees
/// the contiguous borrowed-handle array.
fn generate_free_object_array(prefix: &str) -> TokenStream2 {
    let fn_pfx = format!("{prefix}_");
    let free_fn = format_ident!("{fn_pfx}free_object_array");

    quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #free_fn(a: ffier::FfierObjectArray) {
            unsafe { ffier::ffier_object_array_free(a) };
        }
    }
}

// ===========================================================================
// Strerror bridge generation (batch-level, all errors)
// ===========================================================================

/// Generate `{prefix}_result_name()` and `{prefix}_result_name_cstr()`.
///
/// These decode packed `FfierResult` values (type_tag + code) into static
/// variant name strings. Error handle dispatch (destroy, message, code,
/// result) is handled by the Error trait's self-dispatch infrastructure.
fn generate_strerror_bridge(
    prefix: &str,
    errors: &[TokenStream2],
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let fn_pfx = format!("{prefix}_");

    let result_name_cstr_fn = format_ident!("{fn_pfx}result_name_cstr");
    let result_name_fn = format_ident!("{fn_pfx}result_name");

    // Build dispatch arms for result_name (packed FfierResult → static CStr)
    let mut result_name_dispatch_arms = Vec::new();
    for item in errors {
        if let Ok(meta) = syn::parse2::<MetaError>(item.clone()) {
            let type_tag = meta.type_tag;
            let path = &meta.path;
            result_name_dispatch_arms.push(quote! {
                #type_tag => {
                    let code = ffier::ffier_result_code(r);
                    <#path as #lib_crate::FfiError>::static_message(code).as_ptr()
                }
            });
        }
    }

    let unknown_msg_bytes = proc_macro2::Literal::byte_string(b"unknown error\0");

    quote! {
        /// Variant name as a null-terminated C string.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #result_name_cstr_fn(
            r: ffier::FfierResult,
        ) -> *const core::ffi::c_char {
            if r == 0 { return b"success\0".as_ptr() as *const core::ffi::c_char; }
            let type_tag = ffier::ffier_result_type_tag(r);
            match type_tag {
                #(#result_name_dispatch_arms)*
                _ => unsafe {
                    core::ffi::CStr::from_bytes_with_nul_unchecked(#unknown_msg_bytes).as_ptr()
                },
            }
        }

        /// Variant name as `FtStr` (length-prefixed, no strlen needed).
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #result_name_fn(
            r: ffier::FfierResult,
        ) -> ffier::FfierBytes {
            let ptr = unsafe { #result_name_cstr_fn(r) };
            let len = unsafe { core::ffi::CStr::from_ptr(ptr) }.to_bytes().len();
            ffier::FfierBytes { data: ptr as *const u8, len }
        }
    }
}

// ===========================================================================
// Exportable bridge generation
// ===========================================================================

fn generate_exportable_bridge(
    meta: MetaExportable,
    trait_map: &TraitMap,
    error_map: &ErrorMap,
    handle_types: &HashSet<String>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let struct_path = &meta.struct_path;
    let struct_name = &meta.struct_name.to_string();
    let fn_pfx = meta.fn_pfx();

    let struct_lower = camel_to_snake(struct_name);

    // Whether this type has any builder methods with by-value self.
    // When true, ALL by-value self methods receive a pointer-to-handle
    // (void**) in the C ABI, not just the builder methods themselves.
    let is_builder_type = meta
        .methods
        .iter()
        .any(|m| m.is_builder() && m.receiver == MetaReceiver::Value);

    let mut ffi_fns = Vec::new();

    // Method FFI functions
    for m in &meta.methods {
        let ffi_name_str = format!("{}{}", fn_pfx, m.ffi_name());
        let ffi_name = format_ident!("{}", ffi_name_str);
        let method_name = &m.name;

        let has_receiver = m.receiver != MetaReceiver::None;
        let is_mut = m.receiver == MetaReceiver::Mut;
        let is_by_value = m.receiver == MetaReceiver::Value;
        let is_builder = m.is_builder();

        // Single source of truth: the extern "C" fn signature.
        let c_sig = c_signature_for_method(m, &meta.prefix, handle_types, lib_crate);

        // Self access via borrow/consume (instance methods only).
        //
        // For builder by-value methods, the C param is a pointer-to-pointer
        // (FtConfig* = void**). We dereference it first, storing the slot
        // address for write-back later.
        let obj_binding = if has_receiver {
            let type_assert = quote! {
                let __actual = unsafe { ffier::handle_type_tag(handle) };
                let __expected = <#struct_path as #lib_crate::FfiHandle>::TYPE_TAG;
                if __actual != __expected {
                    __ffier_dispatch_panic(
                        #ffi_name_str,
                        <#struct_path as #lib_crate::FfiHandle>::C_HANDLE_NAME,
                        "",
                        __actual,
                    );
                }
            };
            let deref_slot = if is_builder_type && is_by_value {
                // `handle` param is *mut c_void but actually void**.
                // All by-value self methods on builder types use pointer-to-handle
                // in the C ABI (so the caller can pass &builder consistently).
                quote! {
                    let __handle_slot = handle as *mut *mut core::ffi::c_void;
                    let handle = unsafe { *__handle_slot };
                }
            } else {
                quote! {}
            };
            let cast = if is_by_value {
                quote! {
                    ffier::ffier_handle_consume::<#struct_path>(handle)
                }
            } else if is_mut {
                quote! {
                    ffier::ffier_handle_borrow_mut::<#struct_path>(handle)
                }
            } else {
                quote! {
                    ffier::ffier_handle_borrow::<#struct_path>(handle)
                }
            };
            Some(quote! { #deref_slot #type_assert let obj = unsafe { #cast }; })
        } else {
            None
        };

        // Shared param conversion + impl Trait dispatch.
        let cp = match convert_params(&m.params, &c_sig, &ffi_name_str, trait_map, lib_crate) {
            Ok(cp) => cp,
            Err(err) => return err,
        };
        let converted_args = &cp.converted_args;
        let pre_bindings = &cp.pre_bindings;
        let vtable_pre_bindings = &cp.vtable_pre_bindings;

        // Build the method call expression
        let base_method_call = if has_receiver {
            quote! { obj.#method_name(#(#converted_args),*) }
        } else {
            quote! { <#struct_path>::#method_name(#(#converted_args),*) }
        };

        let method_call = wrap_concrete_dispatch(
            base_method_call,
            &cp.concrete_dispatch_params,
            &ffi_name_str,
            lib_crate,
        );

        // Extern fn signature from c_sig (shared across all return variants)
        let sig_names: Vec<_> = c_sig.params.iter().map(|p| &p.name).collect();
        let sig_types: Vec<_> = c_sig.params.iter().map(|p| &p.c_type).collect();
        let sig_ret = &c_sig.ret;

        let builder_ctx = if is_builder {
            Some(BuilderCtx {
                struct_path,
                is_by_value,
            })
        } else {
            None
        };
        let return_body = wrap_return(
            method_call,
            &m.ret,
            &m.rust_ret,
            handle_types,
            error_map,
            builder_ctx.as_ref(),
            lib_crate,
        );

        let cfg_attr = m.cfg_attr();
        ffi_fns.push(quote! {
            #cfg_attr
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #ffi_name(
                #(#sig_names: #sig_types),*
            ) #sig_ret {
                #obj_binding
                #(#vtable_pre_bindings)*
                #(#pre_bindings)*
                #return_body
            }
        });
    }

    // destroy function
    let destroy_name = format_ident!("{fn_pfx}{struct_lower}_destroy");
    let destroy_str = destroy_name.to_string();

    ffi_fns.push(quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #destroy_name(handle: *mut core::ffi::c_void) {
            if !handle.is_null() {
                let __actual = unsafe { ffier::handle_type_tag(handle) };
                let __expected = <#struct_path as #lib_crate::FfiHandle>::TYPE_TAG;
                if __actual != __expected {
                    __ffier_dispatch_panic(
                        #destroy_str,
                        <#struct_path as #lib_crate::FfiHandle>::C_HANDLE_NAME,
                        "",
                        __actual,
                    );
                }
                unsafe { ffier::ffier_handle_drop::<#struct_path>(handle) };
            }
        }
    });

    quote! {
        #(#ffi_fns)*
    }
}

// ===========================================================================
// Error bridge generation
// ===========================================================================

fn generate_error_bridge(_meta: MetaError, _lib_crate: &TokenStream2) -> TokenStream2 {
    // No per-type bridge functions needed — strerror is emitted at batch level.
    quote! {}
}

// ===========================================================================
// Shared param conversion + impl Trait dispatch
// ===========================================================================

/// Info about a single `impl Trait` param, used for dispatch codegen.
struct ImplTraitParam {
    name: syn::Ident,
    dispatch: crate::meta::DispatchMode,
    ref_kind: crate::meta::ImplTraitRefKind,
    trait_name: String,
    /// (name, bridge_type, cfg_predicate) for each concrete dispatch variant.
    variants: Vec<(String, TokenStream2, Option<TokenStream2>)>,
}

/// Intermediate result of converting method/function params into bridge code.
struct ConvertedParams {
    /// Pre-binding statements (e.g. `let __tags_vec = { ... };`)
    pre_bindings: Vec<TokenStream2>,
    /// Vtable dispatch pre-bindings (e.g. `let mut __dyn_box_fruit = ...;`)
    vtable_pre_bindings: Vec<TokenStream2>,
    /// Converted argument expressions to pass to the Rust function call.
    converted_args: Vec<TokenStream2>,
    /// Impl Trait params with resolved effective dispatch — concrete params
    /// (effective_dispatch = false) need nested type-tag matching via
    /// `wrap_concrete_dispatch`.
    concrete_dispatch_params: Vec<ImplTraitParam>,
}

/// Build the body pieces shared between method and free function bridge
/// generation: param conversion, pre-bindings, and impl Trait dispatch.
///
/// `c_sig` is used to look up the `_len` ident for StrSlice params.
/// `ffi_name_str` is the FFI function name (for error messages in dispatch panics).
fn convert_params(
    params: &[MetaParam],
    c_sig: &CExternSignature,
    ffi_name_str: &str,
    trait_map: &TraitMap,
    lib_crate: &TokenStream2,
) -> Result<ConvertedParams, TokenStream2> {
    // Collect all impl Trait params with their dispatch info.
    // If a trait has no entry in trait_map (no concrete implementors),
    // the param still needs vtable dispatch (empty variants list forces
    // vtable mode since concrete dispatch with 0 variants makes no sense).
    let impl_trait_params: Vec<_> = params
        .iter()
        .filter_map(|p| {
            if let MetaParamKind::ImplTrait {
                trait_name,
                dispatch,
                ref_kind,
                ..
            } = &p.kind
            {
                let info = trait_map.get(trait_name);
                Some(ImplTraitParam {
                    name: p.name.clone(),
                    dispatch: *dispatch,
                    ref_kind: *ref_kind,
                    trait_name: trait_name.clone(),
                    variants: info
                        .map(|i| {
                            i.variants
                                .iter()
                                .map(|v| {
                                    (
                                        v.name.clone(),
                                        v.bridge_type.clone(),
                                        v.cfg_predicate.clone(),
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                })
            } else {
                None
            }
        })
        .collect();

    // Check for dispatch limit (auto mode only)
    let concrete_params: Vec<_> = impl_trait_params
        .iter()
        .filter(|p| p.dispatch != crate::meta::DispatchMode::Vtable)
        .collect();
    let total_branches: u64 = concrete_params
        .iter()
        .map(|p| p.variants.len() as u64)
        .product();
    if total_branches > crate::meta::DEFAULT_MAX_DISPATCH
        && impl_trait_params
            .iter()
            .any(|p| p.dispatch == crate::meta::DispatchMode::Auto)
    {
        let msg = format!(
            "ffier: `{ffi_name_str}` would generate {total_branches} dispatch \
             branches (limit: {}). Add `#[ffier(dispatch = vtable)]` to the impl Trait \
             param(s) or `#[ffier(dispatch = concrete)]` to override the limit.",
            crate::meta::DEFAULT_MAX_DISPATCH,
        );
        return Err(quote! { compile_error!(#msg); });
    }

    // Check vtable dispatch is possible (trait must be exported)
    for p in &impl_trait_params {
        if p.dispatch == crate::meta::DispatchMode::Vtable
            && trait_map
                .get(&p.trait_name)
                .and_then(|info| info.implementable.as_ref())
                .is_none()
        {
            let msg = format!(
                "ffier: `#[ffier(dispatch = vtable)]` on param `{}` requires trait `{}` \
                 to have `#[ffier::export]`",
                p.name, p.trait_name,
            );
            return Err(quote! { compile_error!(#msg); });
        }
    }

    // Convert params
    let mut pre_bindings = Vec::new();
    let converted_args: Vec<_> = params
        .iter()
        .map(|p| {
            let id = &p.name;
            match &p.kind {
                MetaParamKind::ImplTrait { .. } => quote! { #id },
                MetaParamKind::StrSlice | MetaParamKind::HandleSlice(_) => {
                    let len_name = format!("{}_len", p.name);
                    let len_id = &c_sig
                        .params
                        .iter()
                        .find(|cp| cp.name == len_name)
                        .expect("Slice param must have _len param in c_sig")
                        .name;
                    let binding = meta_param_conversion(id, &p.kind, Some(len_id), lib_crate);
                    let vec_id = format_ident!("__{id}_vec");
                    pre_bindings.push(quote! { let #vec_id = #binding; });
                    quote! { &#vec_id }
                }
                MetaParamKind::PrimitiveSlice(_) => {
                    let len_name = format!("{}_len", p.name);
                    let len_id = &c_sig
                        .params
                        .iter()
                        .find(|cp| cp.name == len_name)
                        .expect("Slice param must have _len param in c_sig")
                        .name;
                    let binding = meta_param_conversion(id, &p.kind, Some(len_id), lib_crate);
                    let slice_id = format_ident!("__{id}_slice");
                    pre_bindings.push(quote! { let #slice_id = #binding; });
                    // Already &[T] from from_raw_parts, pass directly.
                    quote! { #slice_id }
                }
                MetaParamKind::Regular(tp) if tp.foreign_crate.is_some() => {
                    let binding = meta_param_conversion(id, &p.kind, None, lib_crate);
                    let foreign_id = format_ident!("__{id}_foreign");
                    pre_bindings.push(binding);
                    // Check if the bridge_type is a reference — if so, borrow the
                    // ManuallyDrop binding. If by-value, pass the binding directly.
                    match syn::parse2::<syn::Type>(tp.bridge_type.clone()).ok() {
                        Some(syn::Type::Reference(r)) if r.mutability.is_some() => {
                            quote! { &mut *#foreign_id }
                        }
                        Some(syn::Type::Reference(_)) => {
                            quote! { &*#foreign_id }
                        }
                        _ => quote! { #foreign_id },
                    }
                }
                other => meta_param_conversion(id, other, None, lib_crate),
            }
        })
        .collect();

    // Determine effective dispatch mode for each param.
    // When total_branches == 0 (no concrete implementors), force vtable
    // dispatch — there's nothing to dispatch concretely against.
    let all_concrete = total_branches > 0
        && impl_trait_params
            .iter()
            .all(|p| p.dispatch != crate::meta::DispatchMode::Vtable)
        && (total_branches <= crate::meta::DEFAULT_MAX_DISPATCH
            || impl_trait_params
                .iter()
                .all(|p| p.dispatch == crate::meta::DispatchMode::Concrete));
    let effective_dispatch: Vec<bool> = if all_concrete {
        vec![false; impl_trait_params.len()]
    } else {
        let mut first_auto_seen = false;
        impl_trait_params
            .iter()
            .map(|p| match p.dispatch {
                crate::meta::DispatchMode::Concrete => false,
                crate::meta::DispatchMode::Vtable => true,
                crate::meta::DispatchMode::Auto => {
                    if !first_auto_seen {
                        first_auto_seen = true;
                        false
                    } else {
                        true
                    }
                }
            })
            .collect()
    };

    // Dynamic dispatch: wrap each vtable-mode param into a dyn Trait reference.
    // For by-value params, consume the handle into Box<dyn Trait> and rebind as &mut.
    // For &/&mut params, borrow the handle directly.
    let mut vtable_pre_bindings: Vec<TokenStream2> = Vec::new();
    for (i, p) in impl_trait_params.iter().enumerate() {
        if !effective_dispatch[i] {
            continue;
        }
        let dyn_id = &p.name;
        let info = trait_map.get(&p.trait_name).unwrap();

        let trait_ident = if let Some(imp) = &info.implementable {
            imp.trait_path.clone()
        } else {
            let ident = format_ident!("{}", p.trait_name);
            quote! { #ident }
        };

        let expected_msg = format!("impl {}", p.trait_name);
        let accepted_const = format_ident!("__FFIER_ACCEPTED_{}", p.trait_name);

        use crate::meta::ImplTraitRefKind;
        match p.ref_kind {
            ImplTraitRefKind::Value => {
                let dyn_box_id = format_ident!("__dyn_box_{}", p.name);
                let mut branches = Vec::new();
                for v in &info.variants {
                    let ty = &v.bridge_type;
                    let vcfg = cfg_predicate_attr(v.cfg_predicate.as_ref());
                    branches.push(quote! {
                        #vcfg
                        if __type_tag == <#ty as #lib_crate::FfiHandle>::TYPE_TAG {
                            let __val = unsafe { ffier::ffier_handle_consume::<#ty>(#dyn_id) };
                            break 'dispatch (Box::new(__val) as Box<dyn #trait_ident>);
                        }
                    });
                }
                vtable_pre_bindings.push(quote! {
                    let mut #dyn_box_id: Box<dyn #trait_ident> = 'dispatch: {
                        let __type_tag = unsafe { ffier::handle_type_tag(#dyn_id) };
                        #(#branches)*
                        __ffier_dispatch_panic(#ffi_name_str, #expected_msg, #accepted_const, __type_tag);
                    };
                    let #dyn_id: &mut dyn #trait_ident = &mut *#dyn_box_id;
                });
            }
            ImplTraitRefKind::Mut => {
                let borrow_id = format_ident!("__borrow_mut_{}", p.name);
                let mut branches = Vec::new();
                for v in &info.variants {
                    let ty = &v.bridge_type;
                    let vcfg = cfg_predicate_attr(v.cfg_predicate.as_ref());
                    branches.push(quote! {
                        #vcfg
                        if __type_tag == <#ty as #lib_crate::FfiHandle>::TYPE_TAG {
                            break 'dispatch (unsafe { ffier::ffier_handle_borrow_mut::<#ty>(#dyn_id) as &mut dyn #trait_ident });
                        }
                    });
                }
                vtable_pre_bindings.push(quote! {
                    let #borrow_id: &mut dyn #trait_ident = 'dispatch: {
                        let __type_tag = unsafe { ffier::handle_type_tag(#dyn_id) };
                        #(#branches)*
                        __ffier_dispatch_panic(#ffi_name_str, #expected_msg, #accepted_const, __type_tag);
                    };
                    let #dyn_id: &mut dyn #trait_ident = #borrow_id;
                });
            }
            ImplTraitRefKind::Ref => {
                let borrow_id = format_ident!("__borrow_{}", p.name);
                let mut branches = Vec::new();
                for v in &info.variants {
                    let ty = &v.bridge_type;
                    let vcfg = cfg_predicate_attr(v.cfg_predicate.as_ref());
                    branches.push(quote! {
                        #vcfg
                        if __type_tag == <#ty as #lib_crate::FfiHandle>::TYPE_TAG {
                            break 'dispatch (unsafe { ffier::ffier_handle_borrow::<#ty>(#dyn_id) as &dyn #trait_ident });
                        }
                    });
                }
                vtable_pre_bindings.push(quote! {
                    let #borrow_id: &dyn #trait_ident = 'dispatch: {
                        let __type_tag = unsafe { ffier::handle_type_tag(#dyn_id) };
                        #(#branches)*
                        __ffier_dispatch_panic(#ffi_name_str, #expected_msg, #accepted_const, __type_tag);
                    };
                    let #dyn_id: &dyn #trait_ident = #borrow_id;
                });
            }
        }
    }

    // Collect concrete dispatch params (effective_dispatch = false)
    let concrete_dispatch_params: Vec<_> = impl_trait_params
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !effective_dispatch[*i])
        .map(|(_, p)| p)
        .collect();

    Ok(ConvertedParams {
        pre_bindings,
        vtable_pre_bindings,
        converted_args,
        concrete_dispatch_params,
    })
}

/// Wrap a base call expression in concrete impl Trait dispatch (nested
/// type-tag matches for non-vtable impl Trait params).
fn wrap_concrete_dispatch(
    base_call: TokenStream2,
    concrete_params: &[ImplTraitParam],
    ffi_name_str: &str,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    concrete_params
        .iter()
        .rev()
        .fold(base_call, |inner, p| {
            let dyn_id = &p.name;
            let variants = &p.variants;
            let label = syn::Lifetime::new(
                &format!("'__dispatch_{}", p.name),
                proc_macro2::Span::call_site(),
            );
            use crate::meta::ImplTraitRefKind;
            let if_branches: Vec<_> = variants
                .iter()
                .map(|(_, ty_tokens, vcfg)| {
                    let vcfg_attr = cfg_predicate_attr(vcfg.as_ref());
                    let binding = match p.ref_kind {
                        ImplTraitRefKind::Value => quote! {
                            let #dyn_id = unsafe { ffier::ffier_handle_consume::<#ty_tokens>(#dyn_id) };
                        },
                        ImplTraitRefKind::Mut => quote! {
                            let #dyn_id = unsafe { ffier::ffier_handle_borrow_mut::<#ty_tokens>(#dyn_id) };
                        },
                        ImplTraitRefKind::Ref => quote! {
                            let #dyn_id = unsafe { ffier::ffier_handle_borrow::<#ty_tokens>(#dyn_id) };
                        },
                    };
                    quote! {
                        #vcfg_attr
                        if __type_tag == <#ty_tokens as #lib_crate::FfiHandle>::TYPE_TAG {
                            #binding
                            break #label { #inner };
                        }
                    }
                })
                .collect();

            let expected_msg = format!("impl {}", p.trait_name);
            let accepted_const = format_ident!("__FFIER_ACCEPTED_{}", p.trait_name);

            quote! { #label: {
                let __type_tag = unsafe { ffier::handle_type_tag(#dyn_id) };
                #(#if_branches)*
                __ffier_dispatch_panic(#ffi_name_str, #expected_msg, #accepted_const, __type_tag);
            }}
        })
}

// ===========================================================================
// Free function bridge generation
// ===========================================================================

fn generate_free_fn_bridge(
    meta: MetaFreeFunction,
    error_map: &ErrorMap,
    handle_types: &HashSet<String>,
    trait_map: &TraitMap,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let fn_path = &meta.fn_path;
    let fn_pfx = meta.fn_pfx();

    // A free function has exactly one "method" in its methods list.
    let m = &meta.methods[0];
    let ffi_name_str = format!("{}{}", fn_pfx, meta.ffi_name);
    let ffi_name = format_ident!("{}", ffi_name_str);

    // Use the same signature builder as methods.
    let c_sig = c_signature_for_method(m, &meta.prefix, handle_types, lib_crate);

    // Shared param conversion + impl Trait dispatch.
    let cp = match convert_params(&m.params, &c_sig, &ffi_name_str, trait_map, lib_crate) {
        Ok(cp) => cp,
        Err(err) => return err,
    };

    let converted_args = &cp.converted_args;
    let base_method_call = quote! { #fn_path(#(#converted_args),*) };
    let method_call = wrap_concrete_dispatch(
        base_method_call,
        &cp.concrete_dispatch_params,
        &ffi_name_str,
        lib_crate,
    );

    let sig_names: Vec<_> = c_sig.params.iter().map(|p| &p.name).collect();
    let sig_types: Vec<_> = c_sig.params.iter().map(|p| &p.c_type).collect();
    let sig_ret = &c_sig.ret;
    let pre_bindings = &cp.pre_bindings;
    let vtable_pre_bindings = &cp.vtable_pre_bindings;

    let return_body = wrap_return(
        method_call,
        &m.ret,
        &m.rust_ret,
        handle_types,
        error_map,
        None,
        lib_crate,
    );

    let cfg_attr = m.cfg_attr();
    quote! {
        #cfg_attr
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #ffi_name(
            #(#sig_names: #sig_types),*
        ) #sig_ret {
            #(#vtable_pre_bindings)*
            #(#pre_bindings)*
            #return_body
        }
    }
}

// ===========================================================================
// Implementable bridge generation
// ===========================================================================

fn generate_implementable_bridge(
    _meta: MetaImplementable,
    _lib_crate: &TokenStream2,
) -> TokenStream2 {
    // No per-type bridge functions needed for exported traits.
    // The vtable ABI is defined by the vtable struct layout, not by generated bridge code.
    quote! {}
}

// ===========================================================================
// Shared C ABI type resolution — used by bridge generation and client generators
// ===========================================================================

/// Check if a type (parsed from rust_ret tokens) is `&T` or `&mut T` where
/// `T` is a handle type. Used to emit `ffier_handle_new_borrowed` for
/// borrowed handle returns.
fn is_borrowed_handle(ty: &syn::Type, handle_types: &HashSet<String>) -> bool {
    if let syn::Type::Reference(ref_ty) = ty {
        crate::meta::type_last_name(&ref_ty.elem)
            .map(|name| name == "Self" || handle_types.contains(&name))
            .unwrap_or(false)
    } else {
        false
    }
}

// ===========================================================================
// Shared return-value conversion
// ===========================================================================

/// Optional builder context for methods that consume `self` and write
/// the new handle back through a double pointer.
struct BuilderCtx<'a> {
    struct_path: &'a TokenStream2,
    is_by_value: bool,
}

/// Wrap a call expression in the appropriate return-value conversion.
///
/// This is the single source of truth for "given an expression that evaluates
/// to the Rust return type, produce tokens that convert it to the C return".
/// Used by exported methods, free functions, and trait dispatch.
fn wrap_return(
    call_expr: TokenStream2,
    ret: &MetaReturn,
    rust_ret: &TokenStream2,
    handle_types: &HashSet<String>,
    error_map: &ErrorMap,
    builder: Option<&BuilderCtx>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    match ret {
        MetaReturn::Void => {
            if let Some(b) = builder.filter(|b| b.is_by_value) {
                // Builder by-value void: method returns Self, write back new handle.
                let struct_path = b.struct_path;
                quote! {
                    let result = #call_expr;
                    let __new_ptr = <#struct_path as #lib_crate::FfiType>::into_c(result);
                    unsafe { *__handle_slot = __new_ptr };
                }
            } else {
                quote! { #call_expr; }
            }
        }
        MetaReturn::Value(tp) => {
            let bridge_type = &tp.bridge_type;
            let tc = resolve_trait_crate(tp, lib_crate);
            let rust_ty = syn::parse2::<syn::Type>(rust_ret.clone()).ok();
            // Foreign types are always handles.
            let is_foreign = tp.foreign_crate.is_some();
            if !is_foreign
                && rust_ty
                    .as_ref()
                    .is_some_and(|ty| is_borrowed_handle(ty, handle_types))
            {
                // &T where T is a handle — shallow-copy into a borrowed handle.
                // Strip the reference to get the inner bridge_type for the
                // FfiHandle::TYPE_TAG lookup.
                let inner_bridge = if let Ok(syn::Type::Reference(ref_ty)) =
                    syn::parse2::<syn::Type>(bridge_type.clone())
                {
                    let elem = &ref_ty.elem;
                    quote! { #elem }
                } else {
                    bridge_type.clone()
                };
                quote! {
                    let result = #call_expr;
                    unsafe {
                        ffier::ffier_handle_new_borrowed::<#inner_bridge>(
                            <#inner_bridge as #tc::FfiHandle>::TYPE_TAG,
                            result as *const #inner_bridge,
                        )
                    }
                }
            } else {
                quote! {
                    let result = #call_expr;
                    <#bridge_type as #tc::FfiType>::into_c(result)
                }
            }
        }
        MetaReturn::HandleSlice { types: tp, direct } => {
            let bridge_type = &tp.bridge_type;
            let tc = resolve_trait_crate(tp, lib_crate);
            // &[&T] or &[T] → build a contiguous Box<[FfierBorrowedHandle]>
            // from the iterator, leak it, return as FfierObjectArray.
            // For &[&T], item is &&T so we deref once; for &[T], item is &T directly.
            let ptr_expr = if *direct {
                quote! { item as *const #bridge_type as *const core::ffi::c_void }
            } else {
                quote! { *item as *const #bridge_type as *const core::ffi::c_void }
            };
            quote! {
                let __slice = #call_expr;
                if __slice.is_empty() {
                    ffier::FfierObjectArray::EMPTY
                } else {
                    let __tag = <#bridge_type as #tc::FfiHandle>::TYPE_TAG;
                    let __meta = ffier::METADATA_BORROWED | ffier::METADATA_ARRAY_ELEMENT;
                    let __boxed: Box<[ffier::FfierBorrowedHandle]> = __slice.iter()
                        .map(|item| ffier::FfierBorrowedHandle {
                            type_tag: __tag,
                            metadata: __meta,
                            ptr: #ptr_expr,
                        })
                        .collect();
                    let __len = __boxed.len();
                    let __raw = Box::into_raw(__boxed) as *const ffier::FfierBorrowedHandle;
                    ffier::FfierObjectArray::from_raw(__raw, __len)
                }
            }
        }
        MetaReturn::Result { ok, err_ident } => {
            // A foreign ok type is always a handle.
            let ok_is_foreign = ok.as_ref().is_some_and(|tp| tp.foreign_crate.is_some());
            let ok_is_handle =
                ok_is_foreign || (ok.is_some() && is_result_ok_handle(rust_ret, handle_types));

            let Some(err_info) = error_map.get(err_ident) else {
                let msg = format!(
                    "method returns Result<_, {err_ident}> but `{err_ident}` is not registered \
                     as an error type. Add `#[derive(ffier::FfiError)]` to `{err_ident}`, or wrap \
                     the error in a type that derives it."
                );
                return quote! { compile_error!(#msg); };
            };
            let err_type_tag = err_info.type_tag;

            let box_expr = quote! {
                if !err_out.is_null() {
                    unsafe { *err_out = <_ as #lib_crate::FfiType>::into_c(e); }
                }
            };

            // Check if the Ok type is a borrowed handle (&HandleType).
            let ok_is_borrowed_handle = !ok_is_foreign && ok.is_some() && {
                let ok_tokens = extract_result_ok_type(rust_ret);
                syn::parse2::<syn::Type>(ok_tokens)
                    .ok()
                    .is_some_and(|ty| is_borrowed_handle(&ty, handle_types))
            };

            // Resolve the trait crate for the ok type (foreign or local).
            let ok_tc = ok
                .as_ref()
                .map(|tp| resolve_trait_crate(tp, lib_crate))
                .unwrap_or(lib_crate);

            if ok_is_handle && !ok_is_borrowed_handle {
                // Return owned handle directly, NULL on error.
                quote! {
                    match #call_expr {
                        Ok(ok_val) => <_ as #ok_tc::FfiType>::into_c(ok_val),
                        Err(e) => { #box_expr core::ptr::null_mut() }
                    }
                }
            } else if ok_is_borrowed_handle {
                // Borrowed handle return: Result<&Handle, E> → new_borrowed, NULL on error.
                let ok_pair = ok.as_ref().unwrap();
                let inner_bridge = if let Ok(syn::Type::Reference(ref_ty)) =
                    syn::parse2::<syn::Type>(ok_pair.bridge_type.clone())
                {
                    let elem = &ref_ty.elem;
                    quote! { #elem }
                } else {
                    ok_pair.bridge_type.clone()
                };
                quote! {
                    match #call_expr {
                        Ok(ok_val) => unsafe {
                            ffier::ffier_handle_new_borrowed::<#inner_bridge>(
                                <#inner_bridge as #ok_tc::FfiHandle>::TYPE_TAG,
                                ok_val as *const #inner_bridge,
                            )
                        },
                        Err(e) => { #box_expr core::ptr::null_mut() }
                    }
                }
            } else {
                let ok_branch = match ok {
                    Some(tp) => {
                        let bridge_type = &tp.bridge_type;
                        quote! {
                            Ok(ok_val) => {
                                unsafe { result.write(<#bridge_type as #ok_tc::FfiType>::into_c(ok_val)) };
                                ffier::FFIER_RESULT_SUCCESS
                            }
                        }
                    }
                    None if builder.is_some_and(|b| b.is_by_value) => {
                        let struct_path = builder.unwrap().struct_path;
                        quote! {
                            Ok(new_self) => {
                                let __new_ptr = <#struct_path as #lib_crate::FfiType>::into_c(new_self);
                                unsafe { *__handle_slot = __new_ptr };
                                ffier::FFIER_RESULT_SUCCESS
                            }
                        }
                    }
                    None => quote! {
                        Ok(_) => ffier::FFIER_RESULT_SUCCESS,
                    },
                };

                let err_branch = if builder.is_some_and(|b| b.is_by_value) {
                    quote! {
                        Err(e) => {
                            let __r = ffier::ffier_result(#err_type_tag, #lib_crate::FfiError::code(&e));
                            unsafe { *__handle_slot = core::ptr::null_mut() };
                            #box_expr
                            __r
                        }
                    }
                } else {
                    quote! {
                        Err(e) => {
                            let __r = ffier::ffier_result(#err_type_tag, #lib_crate::FfiError::code(&e));
                            #box_expr
                            __r
                        }
                    }
                };

                quote! {
                    match #call_expr {
                        #ok_branch
                        #err_branch
                    }
                }
            }
        }
    }
}

/// A single parameter in a C extern signature.
struct CExternParam {
    name: syn::Ident,
    c_type: TokenStream2,
}

/// Complete C extern function signature for a method.
///
/// Contains all the information needed to emit an `unsafe extern "C" { fn ... }`
/// declaration. The parameters include handle, regular params, and out-param
/// (for Result returns) in the order they appear.
struct CExternSignature {
    /// Fully qualified extern function name (e.g. "mylib_calculator_add").
    _fn_name: String,
    /// All parameters in declaration order.
    params: Vec<CExternParam>,
    /// Return type tokens (empty for void).
    ret: TokenStream2,
}

/// Compute the full C extern signature for a method.
///
/// This is the single source of truth for "what does this method look like
/// as an `extern "C"` function".
fn c_signature_for_method(
    method: &MetaMethod,
    prefix: &str,
    handle_types: &HashSet<String>,
    lib_crate: &TokenStream2,
) -> CExternSignature {
    let fn_name = format!("{}_{}", prefix, method.ffi_name());
    let mut params = Vec::new();

    // Handle param (receiver)
    let has_receiver = method.receiver != MetaReceiver::None;

    if has_receiver {
        // All handles are heap-allocated, passed as *mut c_void.
        // Builder by-value methods take *mut *mut c_void (pointer to the
        // caller's handle variable) so the bridge can swap the pointer.
        params.push(CExternParam {
            name: format_ident!("handle"),
            c_type: quote! { *mut core::ffi::c_void },
        });
    }

    // Regular params
    for p in &method.params {
        if matches!(
            p.kind,
            MetaParamKind::StrSlice
                | MetaParamKind::HandleSlice(_)
                | MetaParamKind::PrimitiveSlice(_)
        ) {
            params.push(CExternParam {
                name: p.name.clone(),
                c_type: c_param_type(&p.kind, lib_crate),
            });
            params.push(CExternParam {
                name: format_ident!("{}_len", p.name),
                c_type: quote! { usize },
            });
        } else {
            params.push(CExternParam {
                name: p.name.clone(),
                c_type: c_param_type(&p.kind, lib_crate),
            });
        }
    }

    // Return type + out-param for handle returns or Result
    let ret = match &method.ret {
        MetaReturn::Void => quote! {},
        MetaReturn::Value(_vk) => {
            // All values (handles and primitives) returned directly.
            // Handles return *mut c_void, primitives return their CRepr.
            let ty = c_return_type(_vk, lib_crate);
            quote! { -> #ty }
        }
        MetaReturn::HandleSlice { .. } => {
            // &[&T] or &[T] where T is a handle — return FfierObjectArray by value.
            quote! { -> ffier::FfierObjectArray }
        }
        MetaReturn::Result { ok, .. } => {
            // Foreign ok types are always handles (handle-or-null).
            let ok_is_foreign = ok.as_ref().is_some_and(|tp| tp.foreign_crate.is_some());
            let ok_is_handle = ok_is_foreign
                || (ok.is_some() && is_result_ok_handle(&method.rust_ret, handle_types));
            let ok_is_borrowed_handle = !ok_is_foreign && ok.is_some() && {
                let ok_tokens = extract_result_ok_type(&method.rust_ret);
                syn::parse2::<syn::Type>(ok_tokens)
                    .ok()
                    .is_some_and(|ty| is_borrowed_handle(&ty, handle_types))
            };

            // Builder by-value Result<Self, E>: the ok value is written back
            // through the double-pointer handle param, not returned. Use
            // FtResult style, not handle-or-null.
            let is_builder_self_result =
                method.is_builder() && method.receiver == MetaReceiver::Value;

            if (ok_is_handle || ok_is_borrowed_handle) && !is_builder_self_result {
                // Handle-or-null: return handle directly (NULL on error).
                // err_out is *mut *mut c_void (pointer to caller's FtError variable).
                params.push(CExternParam {
                    name: format_ident!("err_out"),
                    c_type: quote! { *mut *mut core::ffi::c_void },
                });
                quote! { -> *mut core::ffi::c_void }
            } else {
                // FtResult style: return packed error code.
                // Builder-self-result writes ok back through the handle
                // double-pointer, so no separate result out-param.
                if !is_builder_self_result && let Some(vk) = ok {
                    params.push(CExternParam {
                        name: format_ident!("result"),
                        c_type: c_out_param_type(vk, lib_crate),
                    });
                }
                // err_out is *mut *mut c_void (pointer to caller's FtError variable).
                params.push(CExternParam {
                    name: format_ident!("err_out"),
                    c_type: quote! { *mut *mut core::ffi::c_void },
                });
                quote! { -> ffier::FfierResult }
            }
        }
    };

    CExternSignature {
        _fn_name: fn_name,
        params,
        ret,
    }
}

/// Resolve the crate whose `FfiType`/`FfiHandle` traits to use for a type pair.
/// Returns `foreign_crate` if set, otherwise falls back to `lib_crate`.
fn resolve_trait_crate<'a>(tp: &'a MetaTypePair, lib_crate: &'a TokenStream2) -> &'a TokenStream2 {
    tp.foreign_crate.as_ref().unwrap_or(lib_crate)
}

/// Produce the C type tokens for a parameter kind.
fn c_param_type(kind: &MetaParamKind, lib_crate: &TokenStream2) -> TokenStream2 {
    match kind {
        MetaParamKind::Regular(tp) => {
            let bridge_type = &tp.bridge_type;
            let tc = resolve_trait_crate(tp, lib_crate);
            quote! { <#bridge_type as #tc::FfiType>::CRepr }
        }
        MetaParamKind::ImplTrait { .. } => quote! { *mut core::ffi::c_void },
        MetaParamKind::StrSlice => quote! { *const ffier::FfierBytes },
        MetaParamKind::HandleSlice(_) => quote! { *const *mut core::ffi::c_void },
        MetaParamKind::PrimitiveSlice(tp) => {
            let bridge_type = &tp.bridge_type;
            let tc = resolve_trait_crate(tp, lib_crate);
            quote! { *const <#bridge_type as #tc::FfiType>::CRepr }
        }
    }
}

/// Produce the C return type tokens for a value kind.
fn c_return_type(kind: &MetaTypePair, lib_crate: &TokenStream2) -> TokenStream2 {
    let bridge_type = &kind.bridge_type;
    let tc = resolve_trait_crate(kind, lib_crate);
    quote! { <#bridge_type as #tc::FfiType>::CRepr }
}

/// Produce the C type for a Result ok-value out-parameter.
fn c_out_param_type(kind: &MetaTypePair, lib_crate: &TokenStream2) -> TokenStream2 {
    let inner = c_return_type(kind, lib_crate);
    quote! { *mut #inner }
}

fn meta_param_conversion(
    id: &syn::Ident,
    kind: &MetaParamKind,
    len_ident: Option<&syn::Ident>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    match kind {
        MetaParamKind::Regular(tp) if tp.foreign_crate.is_some() => {
            // Foreign handle type from another ffier library's generated client.
            // We can't use FfiType::from_c because the handle's inner value is the
            // real Rust type from the foreign library, not the client wrapper.
            // Instead, wrap the raw pointer in the client handle type directly.
            let tc = tp.foreign_crate.as_ref().unwrap();
            let bridge_type = &tp.bridge_type;
            let foreign_id = format_ident!("__{id}_foreign");

            // Compile-time assert: verify user-provided c_name matches the
            // foreign crate's C_HANDLE_NAME const.
            let c_name_assert = tp.foreign_c_name.as_ref().map(|c_name| {
                let inner = if let Ok(syn::Type::Reference(r)) =
                    syn::parse2::<syn::Type>(bridge_type.clone())
                {
                    let elem = &r.elem;
                    quote! { #elem }
                } else {
                    bridge_type.clone()
                };
                let msg = format!(
                    "foreign c_name mismatch for `{}`: expected \"{}\"",
                    inner, c_name,
                );
                quote! {
                    const _: () = assert!(
                        ffier::const_str_eq(
                            <#inner as #tc::FfiHandle>::C_HANDLE_NAME,
                            #c_name,
                        ),
                        #msg,
                    );
                }
            });

            // Strip the reference (if any) to get the inner handle type.
            if let Ok(syn::Type::Reference(ref_ty)) = syn::parse2::<syn::Type>(bridge_type.clone())
            {
                let inner = &ref_ty.elem;
                if ref_ty.mutability.is_some() {
                    quote! {
                        #c_name_assert
                        let mut #foreign_id = core::mem::ManuallyDrop::new(
                            <#inner as #tc::FfiHandle>::__from_raw(#id));
                    }
                } else {
                    quote! {
                        #c_name_assert
                        let #foreign_id = core::mem::ManuallyDrop::new(
                            <#inner as #tc::FfiHandle>::__from_raw(#id));
                    }
                }
            } else {
                // By-value foreign handle — ownership transfers, Drop is caller's intent.
                quote! {
                    #c_name_assert
                    let #foreign_id = <#bridge_type as #tc::FfiHandle>::__from_raw(#id);
                }
            }
        }
        MetaParamKind::Regular(tp) => {
            let bridge_type = &tp.bridge_type;
            let tc = resolve_trait_crate(tp, lib_crate);
            quote! { unsafe { <#bridge_type as #tc::FfiType>::from_c(#id) } }
        }
        MetaParamKind::StrSlice => {
            let len_id = len_ident.expect("StrSlice conversion needs len_ident");
            quote! { {
                let __slice = unsafe { core::slice::from_raw_parts(#id, #len_id) };
                let __strs: Vec<&str> = __slice.iter().map(|b| unsafe {
                    core::str::from_utf8_unchecked(
                        core::slice::from_raw_parts(b.data, b.len))
                }).collect();
                __strs
            } }
        }
        MetaParamKind::HandleSlice(MetaTypePair { bridge_type, .. }) => {
            let len_id = len_ident.expect("HandleSlice conversion needs len_ident");
            quote! { {
                if #len_id == 0 {
                    Vec::new()
                } else {
                    let __handles = unsafe { core::slice::from_raw_parts(#id, #len_id) };
                    __handles.iter()
                        .map(|h| unsafe { ffier::ffier_handle_borrow::<#bridge_type>(*h) })
                        .collect()
                }
            } }
        }
        MetaParamKind::PrimitiveSlice(_) => {
            let len_id = len_ident.expect("PrimitiveSlice conversion needs len_ident");
            quote! { {
                if #len_id == 0 {
                    &[]
                } else {
                    unsafe { core::slice::from_raw_parts(#id, #len_id) }
                }
            } }
        }
        MetaParamKind::ImplTrait { .. } => {
            quote! { compile_error!("ImplTrait should not use param_conversion") }
        }
    }
}

// ===========================================================================
// Self-dispatch bridge generation
// ===========================================================================

/// Emit a `let obj = ...` binding that borrows a value from a handle.
///
/// - `mutable = false`: `let obj = ffier::ffier_handle_borrow::<T>(handle);`
/// - `mutable = true`:  `let obj = ffier::ffier_handle_borrow_mut::<T>(handle);`
fn borrow_from_handle(ty: &TokenStream2, mutable: bool) -> TokenStream2 {
    if mutable {
        quote! {
            let obj = unsafe { ffier::ffier_handle_borrow_mut::<#ty>(handle) };
        }
    } else {
        quote! {
            let obj = unsafe { ffier::ffier_handle_borrow::<#ty>(handle) };
        }
    }
}

/// Generate per-trait dispatching C functions for an exported trait.
///
/// For each method on the trait, generates a single C function that reads the
/// type tag from the handle and dispatches to the concrete implementor's method.
/// Also generates a dispatching destroy function.
///
/// Example: for `trait Fruit` with method `value(&self) -> i32` and
/// implementors `Apple`, `Orange`, `VtableFruit`:
///
/// ```c
/// int32_t ft_fruit_value(void* handle);
/// void ft_fruit_destroy(void* handle);
/// ```
fn generate_self_dispatch_bridge(
    trait_name: &str,
    info: &TraitDispatchInfo,
    prefix: &str,
    trait_map: &TraitMap,
    error_map: &ErrorMap,
    handle_types: &HashSet<String>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let imp = info
        .implementable
        .as_ref()
        .expect("generate_self_dispatch_bridge called for non-exported trait");
    let trait_path = &imp.trait_path;
    let trait_snake = camel_to_snake(trait_name);
    let fn_pfx = format!("{prefix}_");

    let accepted_const = format_ident!("__FFIER_ACCEPTED_{trait_name}");

    let mut bridge_fns = Vec::new();

    // Generate dispatching functions for each trait method (own methods only,
    // not supertrait methods — those need their own dispatch via their own trait).
    // TODO: Reconsider the `supers(...)` syntax on #[ffier::export] for traits.
    //       Perhaps supertrait methods should be exported by making the supertrait
    //       itself `#[ffier::export]` (or at least having its own trait_impl
    //       entries), rather than inlining them into the subtrait's vtable.
    let own_methods = &imp.methods[..imp.own_method_count];
    for m in own_methods.iter() {
        let method_name = &m.name;
        let ffi_name_str = format!("{fn_pfx}{trait_snake}_{method_name}");
        let ffi_name = format_ident!("{ffi_name_str}");

        // Use the shared signature builder for params + return type.
        // Self-dispatch always needs a `handle` param (even for raw_handle
        // methods that have MetaReceiver::None) because the dispatcher reads
        // the type tag from it. c_signature_for_method only adds handle for
        // methods with a receiver, so prepend it when missing.
        let c_sig = c_signature_for_method(m, prefix, handle_types, lib_crate);
        let has_receiver = m.receiver != MetaReceiver::None;
        let mut all_params: Vec<(&syn::Ident, &TokenStream2)> = Vec::new();
        let handle_name = format_ident!("handle");
        let handle_type = quote! { *mut core::ffi::c_void };
        if !has_receiver {
            all_params.push((&handle_name, &handle_type));
        }
        for p in &c_sig.params {
            all_params.push((&p.name, &p.c_type));
        }
        let sig_names: Vec<_> = all_params.iter().map(|(n, _)| *n).collect();
        let sig_types: Vec<_> = all_params.iter().map(|(_, t)| *t).collect();
        let sig_ret = &c_sig.ret;

        // Shared param conversion (same as exported methods / free functions).
        let cp = match convert_params(&m.params, &c_sig, &ffi_name_str, trait_map, lib_crate) {
            Ok(cp) => cp,
            Err(err) => return err,
        };
        let converted_args = &cp.converted_args;
        let pre_bindings = &cp.pre_bindings;
        let vtable_pre_bindings = &cp.vtable_pre_bindings;

        let method_index_u32 = m.index() as u32;

        // Default helper path for defaulted methods.
        let default_helper_path = if m.has_default() {
            let helper_ident = format_ident!("__ffier_default_{}_{}", trait_name, method_name);
            let mut tokens: Vec<proc_macro2::TokenTree> = trait_path.clone().into_iter().collect();
            if let Some(last) = tokens.last_mut() {
                *last = proc_macro2::TokenTree::Ident(helper_ident);
            }
            Some(tokens.into_iter().collect::<TokenStream2>())
        } else {
            None
        };

        let dispatch_branches: Vec<_> = info
            .variants
            .iter()
            .map(|v| {
                let ty = &v.bridge_type;

                // For the VtableFoo variant of defaulted methods, check metadata
                // before calling through the vtable. Skip for raw_handle methods
                // since exported trait macros don't generate default helpers for them.
                let metadata_guard =
                    if v.kind == TraitVariantKind::Wrapper && m.has_default() && !m.raw_handle() {
                        if let Some(helper) = &default_helper_path {
                            let obj_for_default = borrow_from_handle(ty, m.is_mut());
                            let default_base_call = quote! { #helper(obj #(, #converted_args)*) };
                            let default_call_expr = wrap_concrete_dispatch(
                                default_base_call,
                                &cp.concrete_dispatch_params,
                                &ffi_name_str,
                                lib_crate,
                            );
                            let default_body = wrap_return(
                                default_call_expr,
                                &m.ret,
                                &m.rust_ret,
                                handle_types,
                                error_map,
                                None,
                                lib_crate,
                            );
                            quote! {
                                let __metadata = unsafe { ffier::handle_metadata(handle) };
                                if ffier::default_dispatch_method_index(__metadata)
                                    == Some(#method_index_u32)
                                {
                                    #obj_for_default
                                    return { #default_body };
                                }
                            }
                        } else {
                            quote! {}
                        }
                    } else {
                        quote! {}
                    };

                let (base_call, pre_binding) = if m.raw_handle() {
                    (
                        quote! {
                            <#ty as #trait_path>::#method_name(
                                handle as *const ffier::FfierHandle<#ty> #(, #converted_args)*)
                        },
                        quote! {},
                    )
                } else {
                    let obj_binding = borrow_from_handle(ty, m.is_mut());
                    (
                        quote! {
                            <#ty as #trait_path>::#method_name(obj #(, #converted_args)*)
                        },
                        obj_binding,
                    )
                };
                let call_expr = wrap_concrete_dispatch(
                    base_call,
                    &cp.concrete_dispatch_params,
                    &ffi_name_str,
                    lib_crate,
                );

                let ret_body = wrap_return(
                    call_expr,
                    &m.ret,
                    &m.rust_ret,
                    handle_types,
                    error_map,
                    None,
                    lib_crate,
                );

                let variant_cfg = cfg_predicate_attr(v.cfg_predicate.as_ref());
                quote! {
                    #variant_cfg
                    if __type_tag == <#ty as #lib_crate::FfiHandle>::TYPE_TAG {
                        #metadata_guard
                        #pre_binding
                        return { #ret_body };
                    }
                }
            })
            .collect();

        let expected_str = format!("{trait_name} implementor");
        let cfg_attr = m.cfg_attr();
        bridge_fns.push(quote! {
            #cfg_attr
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #ffi_name(#(#sig_names: #sig_types),*) #sig_ret {
                #(#vtable_pre_bindings)*
                #(#pre_bindings)*
                let __type_tag = unsafe { ffier::handle_type_tag(handle) };
                #(#dispatch_branches)*
                __ffier_dispatch_panic(#ffi_name_str, #expected_str, #accepted_const, __type_tag);
            }
        });
    }

    // Generate dispatching destroy function
    let destroy_name_str = format!("{fn_pfx}{trait_snake}_destroy");
    let destroy_name = format_ident!("{destroy_name_str}");

    let destroy_branches: Vec<_> = info
        .variants
        .iter()
        .map(|v| {
            let ty = &v.bridge_type;
            let variant_cfg = cfg_predicate_attr(v.cfg_predicate.as_ref());
            quote! {
                #variant_cfg
                if __type_tag == <#ty as #lib_crate::FfiHandle>::TYPE_TAG {
                    unsafe { ffier::ffier_handle_drop::<#ty>(handle) };
                    return;
                }
            }
        })
        .collect();

    let destroy_expected = format!("{trait_name} implementor");
    bridge_fns.push(quote! {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn #destroy_name(handle: *mut core::ffi::c_void) {
            if !handle.is_null() {
                let __type_tag = unsafe { ffier::handle_type_tag(handle) };
                #(#destroy_branches)*
                __ffier_dispatch_panic(#destroy_name_str, #destroy_expected, #accepted_const, __type_tag);
            }
        }
    });

    quote! {
        #(#bridge_fns)*
    }
}

// ===========================================================================
// Trait impl bridge generation
// ===========================================================================

fn generate_trait_impl_bridge(
    meta: MetaTraitImpl,
    trait_map: &TraitMap,
    error_map: &ErrorMap,
    handle_types: &HashSet<String>,
    lib_crate: &TokenStream2,
) -> TokenStream2 {
    let struct_path = &meta.struct_path;
    let trait_path = &meta.trait_path;
    let fn_pfx = meta.fn_pfx();
    let struct_name_str = meta.struct_name.to_string();
    let struct_snake = camel_to_snake(&struct_name_str);
    let trait_name_str = meta.trait_name.to_string();
    let trait_snake = camel_to_snake(&trait_name_str);

    // When struct and trait have the same snake_case name (e.g. Error for Error),
    // the per-impl bridge functions would collide with the trait dispatch functions.
    // Skip generating bridge functions — dispatch already handles this case.
    if struct_snake == trait_snake {
        return quote! {};
    }

    let mut bridge_fns = Vec::new();

    for m in &meta.methods {
        let method_name = &m.name;
        let ffi_name_str = format!("{fn_pfx}{struct_snake}_{method_name}");
        let ffi_name = format_ident!("{ffi_name_str}");

        // Use shared signature builder + return wrapper
        let c_sig = c_signature_for_method(m, &meta.prefix, handle_types, lib_crate);
        let sig_names: Vec<_> = c_sig.params.iter().map(|p| &p.name).collect();
        let sig_types: Vec<_> = c_sig.params.iter().map(|p| &p.c_type).collect();
        let sig_ret = &c_sig.ret;

        // Shared param conversion
        let cp = match convert_params(&m.params, &c_sig, &ffi_name_str, trait_map, lib_crate) {
            Ok(cp) => cp,
            Err(err) => return err,
        };
        let converted_args = &cp.converted_args;

        let base_call = if m.raw_handle() {
            quote! {
                <#struct_path as #trait_path>::#method_name(
                    handle as *const ffier::FfierHandle<#struct_path> #(, #converted_args)*)
            }
        } else {
            let borrow = borrow_from_handle(&quote! { #struct_path }, m.is_mut());
            quote! { { #borrow <#struct_path as #trait_path>::#method_name(obj, #(#converted_args),*) } }
        };
        let call_expr = wrap_concrete_dispatch(
            base_call,
            &cp.concrete_dispatch_params,
            &ffi_name_str,
            lib_crate,
        );

        let return_body = wrap_return(
            call_expr,
            &m.ret,
            &m.rust_ret,
            handle_types,
            error_map,
            None,
            lib_crate,
        );

        let pre_bindings = &cp.pre_bindings;
        let vtable_pre_bindings = &cp.vtable_pre_bindings;
        let cfg_attr = m.cfg_attr();
        bridge_fns.push(quote! {
            #cfg_attr
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn #ffi_name(#(#sig_names: #sig_types),*) #sig_ret {
                #(#vtable_pre_bindings)*
                #(#pre_bindings)*
                #return_body
            }
        });
    }

    quote! {
        #(#bridge_fns)*
    }
}

// ===========================================================================
// JSON metadata emission
// ===========================================================================

/// Convert parsed metadata to `ffier_schema::Library` and write to the
/// user-specified `schema_output` path. Relative paths are resolved from
/// `CARGO_MANIFEST_DIR`.
#[allow(clippy::too_many_arguments)]
fn emit_json(
    prefix: &str,
    primitives_prefix: Option<&str>,
    schema_output: &str,
    errors: &[TokenStream2],
    exportables: &[TokenStream2],
    implementables: &[TokenStream2],
    trait_impls: &[TokenStream2],
    enum_constants: &[TokenStream2],
    bitflags_constants: &[TokenStream2],
    free_fns: &[TokenStream2],
) {
    let path = std::path::PathBuf::from(schema_output);
    let path = if path.is_absolute() {
        path
    } else {
        let manifest_dir = match std::env::var("CARGO_MANIFEST_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        manifest_dir.join(path)
    };

    let library = build_schema(
        prefix,
        primitives_prefix,
        errors,
        exportables,
        implementables,
        trait_impls,
        enum_constants,
        bitflags_constants,
        free_fns,
    );
    let json = library.to_json();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    // Only write if contents changed to avoid bumping the timestamp
    // (which would trigger unnecessary downstream rebuilds).
    let needs_write = match std::fs::read(&path) {
        Ok(existing) => existing != json.as_bytes(),
        Err(_) => true,
    };
    if needs_write {
        std::fs::write(&path, json).unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", path.display());
        });
    }
}

/// Context for C type resolution during schema conversion.
struct CTypeResolver {
    type_pfx: String,      // e.g. "Ft"
    prim_type_pfx: String, // e.g. "Ft" (or "Krun" when primitives_prefix differs)
    upper_pfx: String,     // e.g. "FT_"
    fn_pfx: String,        // e.g. "ft_"
}

impl CTypeResolver {
    fn new(prefix: &str, primitives_prefix: Option<&str>) -> Self {
        let type_pfx = crate::meta::snake_to_pascal(prefix);
        let prim_type_pfx = crate::meta::snake_to_pascal(primitives_prefix.unwrap_or(prefix));
        let upper_pfx = format!("{}_", prefix.to_ascii_uppercase());
        let fn_pfx = format!("{prefix}_");
        CTypeResolver {
            type_pfx,
            prim_type_pfx,
            upper_pfx,
            fn_pfx,
        }
    }

    /// Resolve the C type for a handle (always opaque pointer typedef).
    fn handle_c_name(&self, name: &str) -> String {
        format!("{}{}", self.type_pfx, name)
    }

    /// FFI function name: prefix + ffi_name suffix.
    fn ffi_fn_name(&self, ffi_name: &str) -> String {
        format!("{}{}", self.fn_pfx, ffi_name)
    }

    /// Error constant name: FT_ERROR_TEST_NOT_FOUND
    /// Strips "Error" suffix from the error type name: TestError → TEST.
    fn error_const_name(&self, error_name: &str, variant_name: &str) -> String {
        let stripped = error_name.strip_suffix("Error").unwrap_or(error_name);
        let error_upper = camel_to_upper_snake(stripped);
        let variant_upper = camel_to_upper_snake(variant_name);
        format!("{}ERROR_{}_{}", self.upper_pfx, error_upper, variant_upper)
    }

    /// Build a `TypeRef` from a parsed `syn::Type`, extracting references,
    /// lifetimes, `Option<_>`, `Box<str>`, qualified paths, and generic
    /// lifetime arguments structurally — without stringifying and re-parsing.
    fn type_ref_from_syn(&self, ty: &syn::Type) -> ffier_schema::TypeRef {
        match ty {
            // Option<T> → recurse into T and set optional = true.
            syn::Type::Path(tp) if path_is_ident(&tp.path, "Option") => {
                if let Some(inner) = single_angle_arg(&tp.path) {
                    let mut tr = self.type_ref_from_syn(inner);
                    tr.optional = true;
                    return tr;
                }
                // Fallthrough: bare `Option` without angle brackets (shouldn't happen).
                ffier_schema::TypeRef {
                    type_name: "Option".to_string(),
                    ref_kind: ffier_schema::RefKind::None,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                }
            }

            // Box<str> → owned str.
            syn::Type::Path(tp) if path_is_ident(&tp.path, "Box") => {
                if let Some(inner) = single_angle_arg(&tp.path)
                    && matches!(inner, syn::Type::Path(p) if p.path.is_ident("str"))
                {
                    return ffier_schema::TypeRef {
                        type_name: "str".to_string(),
                        ref_kind: ffier_schema::RefKind::None,
                        ref_lifetime: None,
                        type_args: vec![],
                        optional: false,
                        owned: true,
                    };
                }
                // Fallthrough: Box<SomethingElse> — treat like a regular path.
                self.type_ref_from_path(&tp.path, ffier_schema::RefKind::None, None)
            }

            // &T or &mut T — extract reference kind, lifetime, then recurse.
            syn::Type::Reference(ref_ty) => {
                let ref_kind = if ref_ty.mutability.is_some() {
                    ffier_schema::RefKind::Mut
                } else {
                    ffier_schema::RefKind::Shared
                };
                let ref_lifetime = ref_ty.lifetime.as_ref().map(|lt| lt.ident.to_string());
                match ref_ty.elem.as_ref() {
                    syn::Type::Path(tp) => {
                        self.type_ref_from_path(&tp.path, ref_kind, ref_lifetime)
                    }
                    // &[u8] slice — type name is "[u8]".
                    syn::Type::Slice(sl) => {
                        let elem_str = sl.elem.to_token_stream().to_string().replace(' ', "");
                        ffier_schema::TypeRef {
                            type_name: format!("[{elem_str}]"),
                            ref_kind,
                            ref_lifetime,
                            type_args: vec![],
                            optional: false,
                            owned: false,
                        }
                    }
                    // &str — Path("str") should be handled above, but just in case.
                    other => {
                        let name = other.to_token_stream().to_string().replace(' ', "");
                        ffier_schema::TypeRef {
                            type_name: name,
                            ref_kind,
                            ref_lifetime,
                            type_args: vec![],
                            optional: false,
                            owned: false,
                        }
                    }
                }
            }

            // Bare path: i32, str, Widget, View<'a>,
            // ffier_test_foreign_lib::ForeignConfig, etc.
            syn::Type::Path(tp) => {
                self.type_ref_from_path(&tp.path, ffier_schema::RefKind::None, None)
            }

            // *mut c_void / *const c_void → opaque pointer primitives.
            syn::Type::Ptr(ptr) if is_c_void(&ptr.elem) => {
                let type_name = if ptr.mutability.is_some() {
                    ffier_schema::C_VOID_PTR
                } else {
                    ffier_schema::C_VOID_CONST_PTR
                };
                ffier_schema::TypeRef {
                    type_name: type_name.to_string(),
                    ref_kind: ffier_schema::RefKind::None,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                }
            }

            // Other raw pointers — stringify as last resort.
            syn::Type::Ptr(ptr) => {
                let name = ptr.elem.to_token_stream().to_string().replace(' ', "");
                ffier_schema::TypeRef {
                    type_name: name,
                    ref_kind: ffier_schema::RefKind::None,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                }
            }

            // Anything else — stringify as last resort.
            other => {
                let name = other.to_token_stream().to_string().replace(' ', "");
                ffier_schema::TypeRef {
                    type_name: name,
                    ref_kind: ffier_schema::RefKind::None,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                }
            }
        }
    }

    /// Build a `TypeRef` from a `syn::Path`, extracting the last segment
    /// as the type name and any lifetime arguments from angle brackets.
    ///
    /// For qualified paths like `ffier_test_foreign_lib::ForeignConfig`,
    /// this uses the **last segment** (`ForeignConfig`) as the type name —
    /// no spaces-around-`::` nonsense.
    fn type_ref_from_path(
        &self,
        path: &syn::Path,
        ref_kind: ffier_schema::RefKind,
        ref_lifetime: Option<String>,
    ) -> ffier_schema::TypeRef {
        let last = path.segments.last().expect("empty path");
        let type_name = last.ident.to_string();

        // Extract lifetime arguments from angle brackets: View<'a> → ["a"].
        let type_args = match &last.arguments {
            syn::PathArguments::AngleBracketed(ab) => ab
                .args
                .iter()
                .filter_map(|arg| {
                    if let syn::GenericArgument::Lifetime(lt) = arg {
                        Some(lt.ident.to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => vec![],
        };

        ffier_schema::TypeRef {
            type_name,
            ref_kind,
            ref_lifetime,
            type_args,
            optional: false,
            owned: false,
        }
    }

    /// Build a `TypeRef` from a `proc_macro2::TokenStream` by parsing it
    /// as a `syn::Type` first. This is the primary entry point for converting
    /// metadata type tokens to schema type references.
    fn type_ref_from_tokens(&self, tokens: &proc_macro2::TokenStream) -> ffier_schema::TypeRef {
        let ty: syn::Type = syn::parse2(tokens.clone())
            .unwrap_or_else(|e| panic!("failed to parse type tokens `{tokens}`: {e}"));
        self.type_ref_from_syn(&ty)
    }
}

/// Check if a `syn::Path` has a single segment matching `name`
/// (e.g. `Option`, `Box`). Ignores leading `::` and multi-segment paths.
fn path_is_ident(path: &syn::Path, name: &str) -> bool {
    path.segments.len() == 1 && path.segments[0].ident == name
}

/// Check if a `syn::Type` is `c_void` or `core::ffi::c_void` (any qualification).
fn is_c_void(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty {
        let last = tp.path.segments.last();
        last.is_some_and(|seg| seg.ident == "c_void" && seg.arguments.is_none())
    } else {
        false
    }
}

/// Extract the single type argument from `Path<T>` (e.g. `Option<&str>` → `&str`).
/// Returns `None` if there isn't exactly one type argument.
fn single_angle_arg(path: &syn::Path) -> Option<&syn::Type> {
    let last = path.segments.last()?;
    if let syn::PathArguments::AngleBracketed(ab) = &last.arguments
        && ab.args.len() == 1
        && let syn::GenericArgument::Type(ty) = &ab.args[0]
    {
        return Some(ty);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn build_schema(
    prefix: &str,
    primitives_prefix: Option<&str>,
    errors: &[TokenStream2],
    exportables: &[TokenStream2],
    implementables: &[TokenStream2],
    trait_impls: &[TokenStream2],
    enum_constants: &[TokenStream2],
    bitflags_constants: &[TokenStream2],
    free_fns: &[TokenStream2],
) -> ffier_schema::Library {
    let errors_parsed: Vec<_> = errors
        .iter()
        .map(|item| {
            syn::parse2::<MetaError>(item.clone())
                .expect("failed to parse @exported_error metadata")
        })
        .collect();
    let exportables_parsed: Vec<_> = exportables
        .iter()
        .map(|item| {
            syn::parse2::<MetaExportable>(item.clone())
                .expect("failed to parse @exported_impl metadata")
        })
        .collect();
    let implementables_parsed: Vec<_> = implementables
        .iter()
        .map(|item| {
            syn::parse2::<MetaImplementable>(item.clone())
                .expect("failed to parse @exported_trait metadata")
        })
        .collect();
    let trait_impls_parsed: Vec<_> = trait_impls
        .iter()
        .map(|item| {
            syn::parse2::<MetaTraitImpl>(item.clone())
                .expect("failed to parse @exported_trait_impl metadata")
        })
        .collect();
    let enums_parsed: Vec<_> = enum_constants
        .iter()
        .map(|item| {
            syn::parse2::<MetaEnum>(item.clone()).expect("failed to parse @exported_enum metadata")
        })
        .collect();
    let bitflags_parsed: Vec<_> = bitflags_constants
        .iter()
        .map(|item| {
            syn::parse2::<MetaBitflags>(item.clone())
                .expect("failed to parse @exported_bitflags metadata")
        })
        .collect();
    let free_fns_parsed: Vec<_> = free_fns
        .iter()
        .map(|item| {
            syn::parse2::<MetaFreeFunction>(item.clone())
                .expect("failed to parse @exported_fn metadata")
        })
        .collect();

    let resolver = CTypeResolver::new(prefix, primitives_prefix);
    let prim_type_pfx = &resolver.prim_type_pfx;

    // Build handle_types set for Result convention detection.
    let handle_types: HashSet<String> = {
        let mut set = HashSet::new();
        for e in &exportables_parsed {
            set.insert(e.struct_name.to_string());
        }
        for i in &implementables_parsed {
            if !i.no_vtable {
                set.insert(format!("Vtable{}", i.trait_name));
            }
        }
        set
    };

    // Build type registry
    let mut type_registry = BTreeMap::new();

    // Primitives
    for (name, c_type) in &[
        ("i8", "int8_t"),
        ("i16", "int16_t"),
        ("i32", "int32_t"),
        ("i64", "int64_t"),
        ("u8", "uint8_t"),
        ("u16", "uint16_t"),
        ("u32", "uint32_t"),
        ("u64", "uint64_t"),
        ("f32", "float"),
        ("f64", "double"),
        ("isize", "ssize_t"),
        ("usize", "size_t"),
        ("bool", "bool"),
        (ffier_schema::C_VOID_PTR, "void*"),
        (ffier_schema::C_VOID_CONST_PTR, "const void*"),
    ] {
        type_registry.insert(
            name.to_string(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Primitive {
                    c_type: c_type.to_string(),
                },
                type_tag: None,
                bless: None,
                lifetime_params: vec![],
            },
        );
    }

    // Builtins — use primitives prefix for C names
    type_registry.insert(
        "str".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::String {
                c_name: format!("{prim_type_pfx}Str"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::Str),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "[u8]".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Bytes {
                c_name: format!("{prim_type_pfx}Bytes"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::Bytes),
            lifetime_params: vec![],
        },
    );

    // Framework types — ABI scaffolding used by generators.
    type_registry.insert(
        "FfierResult".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: format!("{prim_type_pfx}Result"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::Result),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "FfierPath".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Bytes {
                c_name: format!("{prim_type_pfx}Path"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::Path),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "FfierVtableHandle".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: format!("{prim_type_pfx}VtableHandle"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::VtableHandle),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "FfierObjectArray".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: format!("{prim_type_pfx}ObjectArray"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::ObjectArray),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "FfierObject".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: format!("{prim_type_pfx}Object"),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::Object),
            lifetime_params: vec![],
        },
    );

    // Std type aliases
    type_registry.insert(
        "RawFd".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: "int".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::RawFd),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "BorrowedFd".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Alias {
                alias_of: "RawFd".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::BorrowedFd),
            lifetime_params: vec!["fd".to_string()],
        },
    );
    type_registry.insert(
        "OwnedFd".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Alias {
                alias_of: "RawFd".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::OwnedFd),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "RawHandle".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: "void *".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::RawHandle),
            lifetime_params: vec![],
        },
    );
    type_registry.insert(
        "BorrowedHandle".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Alias {
                alias_of: "RawHandle".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::BorrowedHandle),
            lifetime_params: vec!["handle".to_string()],
        },
    );
    type_registry.insert(
        "OwnedHandle".to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Alias {
                alias_of: "RawHandle".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::OwnedHandle),
            lifetime_params: vec![],
        },
    );

    // Enum constants
    for e in &enums_parsed {
        let name = e.name.to_string();
        type_registry.insert(
            name.clone(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Enum {
                    alias_of: e.repr.clone(),
                },
                type_tag: None,
                bless: None,
                lifetime_params: vec![],
            },
        );
    }

    // Bitflags constants
    for bf in &bitflags_parsed {
        let name = bf.name.to_string();
        type_registry.insert(
            name.clone(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Bitflags {
                    alias_of: bf.repr.clone(),
                },
                type_tag: None,
                bless: None,
                lifetime_params: vec![],
            },
        );
    }

    // Handles (exported types)
    for e in &exportables_parsed {
        let name = e.struct_name.to_string();
        type_registry.insert(
            name.clone(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Handle {
                    c_name: resolver.handle_c_name(&name),
                },
                type_tag: Some(e.type_tag),
                bless: None,
                lifetime_params: e.lifetimes.iter().map(|lt| lt.to_string()).collect(),
            },
        );
    }

    // Errors
    for e in &errors_parsed {
        let name = e.name.to_string();
        type_registry.insert(
            name.clone(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Error {
                    c_name: resolver.handle_c_name(&name),
                },
                type_tag: Some(e.type_tag),
                bless: None,
                lifetime_params: vec![],
            },
        );
    }

    // Implementable traits
    for i in &implementables_parsed {
        let name = i.trait_name.to_string();
        type_registry.insert(
            name.clone(),
            ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Trait {
                    c_name: resolver.handle_c_name(&name),
                },
                type_tag: Some(i.type_tag),
                bless: i.bless.as_deref().map(|b| match b {
                    "error_trait" => ffier_schema::Blessing::ErrorTrait,
                    "push_str" => ffier_schema::Blessing::PushStr,
                    _ => panic!("unknown bless value `{b}` — add a Blessing variant for it"),
                }),
                lifetime_params: i.trait_lifetimes.iter().map(|lt| lt.to_string()).collect(),
            },
        );
    }

    // Traits discovered via trait_impls (no @implementable metadata entry).
    // Infer lifetime params from the trait_lifetime_args of the impls
    // (filtering out 'static which is a concrete binding, not a param).
    for ti in &trait_impls_parsed {
        let name = ti.trait_name.to_string();
        let lifetime_params: Vec<String> = ti
            .trait_lifetime_args
            .iter()
            .filter(|lt| *lt != "static")
            .cloned()
            .collect();
        type_registry
            .entry(name.clone())
            .or_insert_with(|| ffier_schema::TypeEntry {
                kind: ffier_schema::TypeKind::Trait {
                    c_name: resolver.handle_c_name(&name),
                },
                type_tag: None,
                bless: None,
                lifetime_params,
            });
    }

    // Self sentinel — methods returning Self are void at C ABI level.
    type_registry.insert(
        ffier_schema::SELF_TYPE.to_string(),
        ffier_schema::TypeEntry {
            kind: ffier_schema::TypeKind::Primitive {
                c_type: "void".to_string(),
            },
            type_tag: None,
            bless: Some(ffier_schema::Blessing::ReplacesSelf),
            lifetime_params: vec![],
        },
    );

    // Foreign handle types — scan all methods/free-functions for foreign type
    // references and register them so generators know they are opaque handles
    // from another ffier library.
    {
        // Map type_name → (foreign_crate, c_name, lifetime_params).
        let mut foreign_types =
            std::collections::HashMap::<String, (String, Option<String>, Vec<String>)>::new();
        let mut record_foreign = |tp: &MetaTypePair| {
            if let Some(fc) = &tp.foreign_crate {
                let type_ref = resolver.type_ref_from_tokens(&tp.rust_type);
                let type_name = type_ref.type_name;
                let crate_path = fc.to_string().replace(' ', "");
                foreign_types.entry(type_name).or_insert((
                    crate_path,
                    tp.foreign_c_name.clone(),
                    type_ref.type_args,
                ));
            }
        };
        let all_methods = free_fns_parsed
            .iter()
            .flat_map(|f| f.methods.iter())
            .chain(exportables_parsed.iter().flat_map(|e| e.methods.iter()))
            .chain(implementables_parsed.iter().flat_map(|i| i.methods.iter()))
            .chain(trait_impls_parsed.iter().flat_map(|t| t.methods.iter()));
        for m in all_methods {
            for p in &m.params {
                if let MetaParamKind::Regular(tp) = &p.kind {
                    record_foreign(tp);
                }
            }
            match &m.ret {
                MetaReturn::Value(tp) => record_foreign(tp),
                MetaReturn::Result { ok: Some(tp), .. } => record_foreign(tp),
                _ => {}
            }
        }
        for (name, (crate_path, c_name, lifetime_params)) in foreign_types {
            type_registry
                .entry(name.clone())
                .or_insert_with(|| ffier_schema::TypeEntry {
                    kind: ffier_schema::TypeKind::ForeignHandle {
                        foreign_crate: crate_path,
                        c_name: c_name.expect(
                            "foreign type missing c_name (should have been caught earlier)",
                        ),
                    },
                    type_tag: None,
                    bless: None,
                    lifetime_params,
                });
        }
    }

    // Build schema fields while type_registry is still borrowable.
    let exported_types: Vec<_> = exportables_parsed
        .iter()
        .map(|e| convert_exportable(e, &resolver, &handle_types, &type_registry))
        .collect();
    let errors: Vec<_> = errors_parsed
        .iter()
        .map(|e| convert_error(e, &resolver))
        .collect();
    let enum_constants: Vec<_> = enums_parsed
        .iter()
        .map(|e| convert_enum(e, &resolver))
        .collect();
    let bitflags_constants: Vec<_> = bitflags_parsed
        .iter()
        .map(|bf| convert_bitflags(bf, &resolver))
        .collect();
    let free_functions: Vec<_> = free_fns_parsed
        .iter()
        .map(|f| convert_free_fn(f, &resolver, &handle_types, &type_registry))
        .collect();
    let traits: Vec<_> = implementables_parsed
        .iter()
        .map(|i| convert_implementable(i, &resolver, &handle_types, &type_registry))
        .collect();
    let trait_impls: Vec<_> = trait_impls_parsed
        .iter()
        .map(|t| convert_trait_impl(t, &resolver, &handle_types, &type_registry))
        .collect();

    let mut library = ffier_schema::Library {
        prefix: prefix.to_string(),
        primitives_prefix: primitives_prefix.map(|s| s.to_string()),
        type_registry,
        exported_types,
        errors,
        enum_constants,
        bitflags_constants,
        free_functions,
        traits,
        trait_impls,
    };
    library.prune_unreferenced_types();
    library
}

/// Render a cfg predicate to a string for the JSON schema — the schema is a
/// serialized artifact read by separate generator processes, so this is the
/// one legitimate point where the predicate is turned into text.
fn cfg_to_schema_string(cfg: Option<&TokenStream2>) -> Option<String> {
    cfg.map(|ts| ts.to_string())
}

fn convert_enum(meta: &MetaEnum, r: &CTypeResolver) -> ffier_schema::EnumType {
    let name = meta.name.to_string();
    let stripped = name.as_str();
    let name_upper = camel_to_upper_snake(stripped);
    ffier_schema::EnumType {
        name: name.clone(),
        variants: meta
            .variants
            .iter()
            .map(|v| {
                let variant_upper = camel_to_upper_snake(&v.name.to_string());
                ffier_schema::EnumVariant {
                    name: v.name.to_string(),
                    c_name: format!("{}{}_{}", r.upper_pfx, name_upper, variant_upper),
                    value: v.value,
                }
            })
            .collect(),
        cfg: cfg_to_schema_string(meta.cfg_predicate.as_ref()),
    }
}

fn convert_bitflags(meta: &MetaBitflags, r: &CTypeResolver) -> ffier_schema::EnumType {
    let name = meta.name.to_string();
    let name_upper = camel_to_upper_snake(&name);
    ffier_schema::EnumType {
        name: name.clone(),
        variants: meta
            .variants
            .iter()
            .map(|v| {
                // Bitflags constant names are already UPPER_SNAKE_CASE
                // (e.g. READ, WRITE), so use them verbatim — don't run
                // camel_to_upper_snake which would split "READ" into "R_E_A_D".
                let variant_name = v.name.to_string();
                ffier_schema::EnumVariant {
                    name: variant_name.clone(),
                    c_name: format!("{}{}_{}", r.upper_pfx, name_upper, variant_name),
                    value: v.value,
                }
            })
            .collect(),
        cfg: cfg_to_schema_string(meta.cfg_predicate.as_ref()),
    }
}

fn convert_free_fn(
    meta: &MetaFreeFunction,
    r: &CTypeResolver,
    handle_types: &HashSet<String>,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::FreeFunction {
    // A free function has exactly one "method" in its methods list.
    let m = &meta.methods[0];
    // Use type-level cfg_predicate if set, otherwise fall back to method-level cfg.
    let cfg = meta.cfg_predicate.as_ref().or(m.cfg.as_ref());
    ffier_schema::FreeFunction {
        name: meta.name.to_string(),
        ffi_name: r.ffi_fn_name(&meta.ffi_name),
        doc: meta.doc.clone(),
        params: m
            .params
            .iter()
            .map(|p| convert_param(p, r, type_registry))
            .collect(),
        ret: convert_return(&m.ret, &m.rust_ret, r, false, handle_types),
        cfg: cfg_to_schema_string(cfg),
    }
}

fn convert_exportable(
    meta: &MetaExportable,
    r: &CTypeResolver,
    handle_types: &HashSet<String>,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::ExportedType {
    let name = meta.struct_name.to_string();
    let name_snake = camel_to_snake(&name);
    let is_builder_type = meta
        .methods
        .iter()
        .any(|m| m.is_builder() && m.receiver == MetaReceiver::Value);
    ffier_schema::ExportedType {
        name,
        destroy_ffi_name: r.ffi_fn_name(&format!("{name_snake}_destroy")),
        is_builder_type,
        methods: meta
            .methods
            .iter()
            .map(|m| convert_method(m, r, None, handle_types, type_registry))
            .collect(),
        cfg: cfg_to_schema_string(meta.cfg_predicate.as_ref()),
    }
}

fn convert_error(meta: &MetaError, r: &CTypeResolver) -> ffier_schema::ErrorType {
    ffier_schema::ErrorType {
        name: meta.name.to_string(),
        variants: meta
            .variants
            .iter()
            .map(|v| {
                let fields = v
                    .field_types
                    .iter()
                    .map(|ty_tokens| ffier_schema::ErrorField {
                        type_ref: r.type_ref_from_tokens(ty_tokens),
                    })
                    .collect();
                ffier_schema::ErrorVariant {
                    name: v.name.to_string(),
                    c_name: r.error_const_name(&meta.name.to_string(), &v.name.to_string()),
                    code: v.code,
                    message: v.message.clone(),
                    fields,
                }
            })
            .collect(),
    }
}

fn convert_implementable(
    meta: &MetaImplementable,
    r: &CTypeResolver,
    handle_types: &HashSet<String>,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::ImplementableTrait {
    let name = meta.trait_name.to_string();
    let name_snake = camel_to_snake(&name);
    let ffi_prefix = format!("{name_snake}_");
    let name_upper_snake = camel_to_snake(&name).to_ascii_uppercase();
    ffier_schema::ImplementableTrait {
        name: name.clone(),
        destroy_ffi_name: r.ffi_fn_name(&format!("{name_snake}_destroy")),
        type_tag_constant: format!("{}{name_upper_snake}_TYPE_TAG", r.upper_pfx),
        vtable_struct_c_name: format!("{}{}Vtable", r.type_pfx, name),
        wrapper_c_name: format!("{}Vtable{}", r.type_pfx, name),
        vtable_struct_name: format!("{name}Vtable"),
        wrapper_name: format!("Vtable{name}"),
        methods: meta
            .methods
            .iter()
            .map(|m| convert_method(m, r, Some(&ffi_prefix), handle_types, type_registry))
            .collect(),
        own_method_count: meta.own_method_count,
        max_vtable_slot: meta.max_vtable_slot,
        no_vtable: meta.no_vtable,
        cfg: cfg_to_schema_string(meta.cfg_predicate.as_ref()),
    }
}

fn convert_trait_impl(
    meta: &MetaTraitImpl,
    r: &CTypeResolver,
    handle_types: &HashSet<String>,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::TraitImpl {
    ffier_schema::TraitImpl {
        trait_name: meta.trait_name.to_string(),
        struct_name: meta.struct_name.to_string(),
        lifetimes: meta.lifetimes.iter().map(|lt| lt.to_string()).collect(),
        trait_lifetime_args: meta.trait_lifetime_args.clone(),
        struct_lifetime_args: meta.struct_lifetime_args.clone(),
        methods: meta
            .methods
            .iter()
            .map(|m| convert_method(m, r, None, handle_types, type_registry))
            .collect(),
        cfg: cfg_to_schema_string(meta.cfg_predicate.as_ref()),
    }
}

/// Convert a method to its schema representation.
/// `parent_ffi_prefix` is the `"{type_snake}_"` prefix for the parent type/trait
/// (e.g. `"fruit_"` for an exported trait).
/// Only needed for trait definition methods; concrete methods already carry their own ffi_name.
fn convert_method(
    meta: &MetaMethod,
    r: &CTypeResolver,
    parent_ffi_prefix: Option<&str>,
    handle_types: &HashSet<String>,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::Method {
    let (ffi_name, trait_definition) = match &meta.context {
        MetaMethodContext::Exportable { ffi_name, .. } => (r.ffi_fn_name(ffi_name), None),
        MetaMethodContext::Trait {
            has_default, index, ..
        } => {
            let prefix = parent_ffi_prefix.expect("trait method requires parent_ffi_prefix");
            let ffi_name = r.ffi_fn_name(&format!("{prefix}{}", meta.name));
            let td = ffier_schema::TraitMethodDefinition {
                index: *index,
                has_default: *has_default,
            };
            (ffi_name, Some(td))
        }
    };

    let ret = convert_return(
        &meta.ret,
        &meta.rust_ret,
        r,
        meta.is_builder(),
        handle_types,
    );

    ffier_schema::Method {
        name: meta.name.to_string(),
        doc: meta.doc().to_vec(),
        receiver: convert_receiver(meta.receiver),
        method_lifetimes: meta
            .method_lifetimes
            .iter()
            .map(|lt| lt.to_string())
            .collect(),
        params: meta
            .params
            .iter()
            .map(|p| convert_param(p, r, type_registry))
            .collect(),
        ret,
        ffi_name,
        trait_definition,
        cfg: cfg_to_schema_string(meta.cfg.as_ref()),
    }
}

fn convert_receiver(recv: MetaReceiver) -> ffier_schema::Receiver {
    match recv {
        MetaReceiver::None => ffier_schema::Receiver::None,
        MetaReceiver::Ref => ffier_schema::Receiver::Ref,
        MetaReceiver::Mut => ffier_schema::Receiver::Mut,
        MetaReceiver::Value => ffier_schema::Receiver::Value,
    }
}

fn convert_param(
    p: &crate::meta::MetaParam,
    r: &CTypeResolver,
    type_registry: &BTreeMap<String, ffier_schema::TypeEntry>,
) -> ffier_schema::Param {
    let param_type = match &p.kind {
        MetaParamKind::Regular(tp) => {
            ffier_schema::ParamType::Regular(r.type_ref_from_tokens(&tp.rust_type))
        }
        MetaParamKind::StrSlice => {
            // &[&str] → two C params: pointer to FfierBytes array + length.
            // The element type is &str (a reference to str).
            let str_c = format!("{}Str", r.prim_type_pfx);
            ffier_schema::ParamType::Slice {
                element: ffier_schema::TypeRef {
                    type_name: "str".to_string(),
                    ref_kind: ffier_schema::RefKind::Shared,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                },
                c_params: vec![
                    ffier_schema::CParam {
                        name: p.name.to_string(),
                        c_type: format!("const {str_c}*"),
                    },
                    ffier_schema::CParam {
                        name: format!("{}_len", p.name),
                        c_type: "size_t".to_string(),
                    },
                ],
            }
        }
        MetaParamKind::HandleSlice(tp) => {
            // &[&T] → two C params: pointer to handle array + length.
            let elem_ref = r.type_ref_from_tokens(&tp.rust_type);
            let handle_c = r.handle_c_name(&elem_ref.type_name);
            ffier_schema::ParamType::Slice {
                element: ffier_schema::TypeRef {
                    type_name: elem_ref.type_name,
                    ref_kind: ffier_schema::RefKind::Shared,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                },
                c_params: vec![
                    ffier_schema::CParam {
                        name: p.name.to_string(),
                        c_type: format!("const {handle_c}*"),
                    },
                    ffier_schema::CParam {
                        name: format!("{}_len", p.name),
                        c_type: "size_t".to_string(),
                    },
                ],
            }
        }
        MetaParamKind::PrimitiveSlice(tp) => {
            // &[T] where T is a primitive → two C params: pointer to T array + length.
            let elem_ref = r.type_ref_from_tokens(&tp.rust_type);
            let entry = type_registry.get(&elem_ref.type_name).unwrap_or_else(|| {
                panic!(
                    "primitive type `{}` not found in type_registry",
                    elem_ref.type_name
                )
            });
            let elem_c = match &entry.kind {
                ffier_schema::TypeKind::Primitive { c_type } => c_type.as_str(),
                _ => panic!(
                    "expected primitive type for `{}`, got {:?}",
                    elem_ref.type_name, entry.kind
                ),
            };
            ffier_schema::ParamType::Slice {
                element: ffier_schema::TypeRef {
                    type_name: elem_ref.type_name,
                    ref_kind: ffier_schema::RefKind::None,
                    ref_lifetime: None,
                    type_args: vec![],
                    optional: false,
                    owned: false,
                },
                c_params: vec![
                    ffier_schema::CParam {
                        name: p.name.to_string(),
                        c_type: format!("const {elem_c}*"),
                    },
                    ffier_schema::CParam {
                        name: format!("{}_len", p.name),
                        c_type: "size_t".to_string(),
                    },
                ],
            }
        }
        MetaParamKind::ImplTrait {
            trait_name,
            trait_lifetime_args,
            ..
        } => ffier_schema::ParamType::ImplTrait {
            trait_name: trait_name.clone(),
            type_args: trait_lifetime_args
                .iter()
                .map(|lt| lt.to_string())
                .collect(),
        },
    };
    ffier_schema::Param {
        name: p.name.to_string(),
        param_type,
    }
}

fn builder_self_type_ref() -> ffier_schema::TypeRef {
    ffier_schema::TypeRef {
        type_name: ffier_schema::SELF_TYPE.to_string(),
        ref_kind: ffier_schema::RefKind::None,
        ref_lifetime: None,
        type_args: vec![],
        optional: false,
        owned: false,
    }
}

fn convert_return(
    ret: &MetaReturn,
    rust_ret: &TokenStream2,
    r: &CTypeResolver,
    is_builder: bool,
    handle_types: &HashSet<String>,
) -> ffier_schema::Return {
    match ret {
        MetaReturn::Void if is_builder => {
            // `-> Self`: encode as Value(Self).
            ffier_schema::Return::Value(builder_self_type_ref())
        }
        MetaReturn::Void => ffier_schema::Return::Void,
        MetaReturn::Value(tp) => ffier_schema::Return::Value(r.type_ref_from_tokens(&tp.rust_type)),
        MetaReturn::HandleSlice { types, .. } => {
            // &[&T] or &[T] → returns FfierObjectArray with element type info.
            ffier_schema::Return::ObjectArray {
                element: r.type_ref_from_tokens(&types.rust_type),
            }
        }
        MetaReturn::Result { ok, err_ident } if is_builder => {
            // Builder `-> Result<Self, E>`: ok was suppressed to None by
            // annotations; restore it as Self.
            let ok_ref = match ok {
                None => Some(builder_self_type_ref()),
                Some(tp) => Some(r.type_ref_from_tokens(&tp.rust_type)),
            };
            let ok_is_handle = ok.is_some() && is_result_ok_handle(rust_ret, handle_types);
            ffier_schema::Return::Result {
                ok: ok_ref,
                err_type: err_ident.clone(),
                c_convention: if ok_is_handle {
                    ffier_schema::CResultConvention::HandleOrNull
                } else {
                    ffier_schema::CResultConvention::OutParam
                },
            }
        }
        MetaReturn::Result { ok, err_ident } => {
            let ok_ref = ok.as_ref().map(|tp| r.type_ref_from_tokens(&tp.rust_type));
            let ok_is_foreign = ok.as_ref().is_some_and(|tp| tp.foreign_crate.is_some());
            let ok_is_handle =
                ok_is_foreign || (ok.is_some() && is_result_ok_handle(rust_ret, handle_types));
            let ok_is_borrowed_handle = !ok_is_foreign && ok.is_some() && {
                let ok_tokens = extract_result_ok_type(rust_ret);
                syn::parse2::<syn::Type>(ok_tokens)
                    .ok()
                    .is_some_and(|ty| is_borrowed_handle(&ty, handle_types))
            };
            ffier_schema::Return::Result {
                ok: ok_ref,
                err_type: err_ident.clone(),
                c_convention: if ok_is_handle || ok_is_borrowed_handle {
                    ffier_schema::CResultConvention::HandleOrNull
                } else {
                    ffier_schema::CResultConvention::OutParam
                },
            }
        }
    }
}
