use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Data, DeriveInput, FnArg, GenericArgument, ImplItem, ItemImpl, ItemTrait, ItemUse, LitStr, Pat,
    PathArguments, ReturnType, Token, TraitItem, Type, parse::Parse, parse_macro_input,
    visit_mut::VisitMut,
};

mod bridge;
mod meta;

use meta::{camel_to_snake, camel_to_upper_snake, erase_lifetimes};

/// Counter for generating unique `#[macro_export]` macro names.
/// The exported name is an implementation detail — users access the macro
/// only through the `pub use ... as __ffier_meta_*` alias placed next to the type.
static MACRO_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// cfg_attr unwrapping — #[cfg_attr(COND, ffier(...))] → #[ffier(...)]
// ---------------------------------------------------------------------------

/// Extract `#[ffier(...)]` from an attribute — handles both plain
/// `#[ffier(...)]` and `#[cfg_attr(COND, ffier(...))]`.
/// Returns `None` for non-ffier attrs.
fn extract_ffier_attr(attr: &syn::Attribute) -> Option<syn::Attribute> {
    if attr.path().is_ident("ffier") {
        return Some(attr.clone());
    }
    if !attr.path().is_ident("cfg_attr") {
        return None;
    }
    let tokens = match &attr.meta {
        syn::Meta::List(list) => list.tokens.clone(),
        _ => return None,
    };

    // Find the top-level comma separating condition from the inner attr,
    // then check if the remainder starts with `ffier`.
    // Group tokens (parenthesized, braced, bracketed) are atomic in
    // proc_macro2 — a comma inside `all(feature = "ffi")` is inside the
    // group's stream and never appears as a top-level Punct.
    let mut after_comma = proc_macro2::TokenStream::new();
    let mut found_comma = false;

    for tt in tokens {
        if !found_comma {
            if let proc_macro2::TokenTree::Punct(p) = &tt
                && p.as_char() == ','
            {
                found_comma = true;
                continue;
            }
        } else {
            after_comma.extend(std::iter::once(tt));
        }
    }

    if !found_comma {
        return None;
    }

    let mut after_iter = after_comma.clone().into_iter();
    if let Some(proc_macro2::TokenTree::Ident(id)) = after_iter.next()
        && id == "ffier"
    {
        return Some(syn::parse_quote! { #[#after_comma] });
    }

    None
}

/// Strip all ffier attrs (direct or cfg_attr-wrapped) from an attribute list.
fn strip_ffier_attrs(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|a| extract_ffier_attr(a).is_none());
}

// ---------------------------------------------------------------------------
// Type classification for params and return values
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Unified method/param/return types — shared by all annotation macros
// ---------------------------------------------------------------------------

/// A bridge/rust type pair. Used for params and return types.
struct TypePair {
    bridge: proc_macro2::TokenStream,
    rust: proc_macro2::TokenStream,
    /// When set, the type comes from a foreign ffier library and this
    /// is the crate path whose `FfiType`/`FfiHandle` traits should be used.
    foreign_crate: Option<proc_macro2::TokenStream>,
    /// C typedef name for the foreign handle type (e.g. `"FlForeignConfig"`).
    foreign_c_name: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Receiver {
    /// Static method (no receiver).
    None,
    /// `&self`
    Ref,
    /// `&mut self`
    Mut,
    /// `self` (consuming, by value)
    Value,
}

enum ParamKind {
    /// Uniform: bridge_type resolves via `<T as FfiType>::CRepr`.
    Regular,
    /// `&[&str]` — slice of string references, expands to two C params.
    StrSlice,
    /// `&[T]` where T is an exported handle type — slice of struct references,
    /// expands to two C params (pointer to handle array + length).
    HandleSlice,
    /// `&[T]` where T is a primitive type (u32, i64, bool, etc.) —
    /// expands to two C params (pointer to element array + length).
    PrimitiveSlice,
    /// `impl Trait` parameter — generator resolves dispatch types from trait map.
    ImplTrait {
        trait_name: String,
        /// Dispatch mode: "auto", "concrete", or "vtable".
        /// For trait methods this defaults to "auto".
        dispatch: String,
        /// How the param is passed: "value", "ref", or "mut".
        ref_kind: String,
        /// Lifetime arguments on the trait (e.g. `["a"]` for `impl Snapshot<'a>`).
        trait_lifetime_args: Vec<String>,
    },
}

enum ReturnKind {
    Void,
    Value(TypePair),
    Result {
        ok: Option<TypePair>,
        err_ident: String,
    },
    /// `&[&T]` or `&[T]` where T is an exported handle type — returns a
    /// contiguous array of borrowed handles. `direct` is true for `&[T]`.
    HandleSlice {
        types: TypePair,
        direct: bool,
    },
}

struct ParamInfo {
    name: syn::Ident,
    kind: ParamKind,
    /// Type pair for this param. Present for `Regular` and `ImplTrait`,
    /// `None` for `StrSlice` (which expands to two C params).
    types: Option<TypePair>,
}

struct MethodInfo {
    name: syn::Ident,
    receiver: Receiver,
    params: Vec<ParamInfo>,
    ret: ReturnKind,
    // --- exported-impl-specific (defaults for trait methods) ---
    /// FFI function name suffix (e.g. `"widget_new"`). Empty for trait methods.
    ffi_name: String,
    /// True if this method returns Self (builder pattern).
    is_builder: bool,
    /// Method-level lifetime params.
    method_lifetimes: Vec<syn::Ident>,
    doc_lines: Vec<String>,
    /// Original Rust return type for client codegen.
    rust_ret: Option<proc_macro2::TokenStream>,
    // --- trait-specific (defaults for exported impl methods) ---
    /// Whether this method has a default impl in the trait.
    has_default: bool,
    /// Vtable slot index.
    index: usize,
    /// Raw handle method (receives `*const FfierHandle<Self>` instead of `&self`).
    raw_handle: bool,
    /// `#[cfg(...)]` predicates from the method's attributes. When present,
    /// the bridge function is gated behind this predicate and the schema
    /// records it so generators can emit conditional code.
    cfg_predicates: Vec<syn::Meta>,
}

impl ParamInfo {
    fn is_impl_trait(&self) -> bool {
        matches!(self.kind, ParamKind::ImplTrait { .. })
    }
}

/// Detect `&[&str]` — a slice of string references.
fn is_str_slice(ty: &Type) -> bool {
    let Type::Reference(ref_ty) = ty else {
        return false;
    };
    let Type::Slice(sl) = &*ref_ty.elem else {
        return false;
    };
    let Type::Reference(inner_ref) = &*sl.elem else {
        return false;
    };
    matches!(&*inner_ref.elem, Type::Path(tp) if tp.path.is_ident("str"))
}

/// Check for types that should not be used across FFI and return a
/// compile error message if found. Returns `None` if the type is fine.
fn check_forbidden_ffi_type(ty: &Type) -> Option<&'static str> {
    // Peel through references
    let inner = match ty {
        Type::Reference(r) => &*r.elem,
        _ => ty,
    };
    let Type::Path(tp) = inner else { return None };
    let last = tp.path.segments.last()?;
    let name = last.ident.to_string();
    match name.as_str() {
        "String" => {
            return Some(
                "String cannot be used in FFI — use `Box<str>` (owned) or `&str` (borrowed) instead",
            );
        }
        "Vec" => {
            return Some(
                "Vec<T> cannot be used in FFI — use `&[T]` (borrowed slice) or `Box<[T]>` instead",
            );
        }
        _ => {}
    }
    // Recurse into generic args (Result<String, E>, Option<Vec<T>>, etc.)
    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
        for arg in &args.args {
            if let syn::GenericArgument::Type(inner_ty) = arg
                && let Some(msg) = check_forbidden_ffi_type(inner_ty)
            {
                return Some(msg);
            }
        }
    }
    None
}

/// Detect `&[&T]` or `&[T]` where T is a non-primitive named type
/// (i.e. an exported handle type). Returns the element type T if matched.
/// Excludes `&[&str]` (handled as StrSlice).
fn handle_slice_elem(ty: &Type) -> Option<&Type> {
    let Type::Reference(ref_ty) = ty else {
        return None;
    };
    let Type::Slice(sl) = &*ref_ty.elem else {
        return None;
    };
    // Peel off inner & if present: &[&T] → T, &[T] → T
    let elem = match &*sl.elem {
        Type::Reference(inner_ref) => &*inner_ref.elem,
        other => other,
    };
    // Not &[&str] / &[str] (already handled by is_str_slice)
    if matches!(elem, Type::Path(tp) if tp.path.is_ident("str")) {
        return None;
    }
    // Must be a named type that isn't a primitive
    let Type::Path(tp) = elem else {
        return None;
    };
    let name = tp
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    if PRIMITIVES.contains(&name.as_str()) {
        return None;
    }
    Some(elem)
}

/// Compat wrapper — returns true for `&[&T]` or `&[T]` handle slices.
fn is_handle_slice(ty: &Type) -> bool {
    handle_slice_elem(ty).is_some()
}

/// Detect `&[T]` where T is a primitive type (u32, i64, bool, etc.).
/// Returns the element type T if matched.
/// Excludes `&[u8]` (already handled as `FfierBytes`/bytes).
fn primitive_slice_elem(ty: &Type) -> Option<&Type> {
    let Type::Reference(ref_ty) = ty else {
        return None;
    };
    let Type::Slice(sl) = &*ref_ty.elem else {
        return None;
    };
    // Must not have an inner reference — &[T] not &[&T]
    if matches!(&*sl.elem, Type::Reference(_)) {
        return None;
    }
    let Type::Path(tp) = &*sl.elem else {
        return None;
    };
    let name = tp
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    // Must be a primitive, but not u8 (which is already handled as bytes)
    if name == "u8" || !PRIMITIVES.contains(&name.as_str()) {
        return None;
    }
    Some(&*sl.elem)
}

// ---------------------------------------------------------------------------
// Unified export macro
// ---------------------------------------------------------------------------

/// Export a Rust item for FFI consumption.
///
/// Dispatches automatically based on the annotated item:
/// - **`impl Struct { ... }`** — export struct methods as C functions
/// - **`impl Trait for Struct { ... }`** — bridge trait impl methods to C
/// - **`trait Foo { ... }`** — declare a trait implementable from C via vtable
/// - **`enum Foo { ... }`** — export enum variants as C constants
/// - **`fn foo(...) { ... }`** — export a free function to C
///
/// Trait definitions accept optional arguments:
/// ```ignore
/// #[ffier::export(reserved(1, 3), foreign, bless = "error_trait")]
/// trait Foo { ... }
/// ```
///
/// All item kinds also accept `cfg = "predicate"` — see [`ExportCfg`].
#[proc_macro_attribute]
pub fn export(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item2: proc_macro2::TokenStream = item.clone().into();
    let attr2: proc_macro2::TokenStream = attr.clone().into();

    // 1. Trait definition → exported trait
    if let Ok(trait_item) = syn::parse2::<ItemTrait>(item2.clone()) {
        let args = match syn::parse2::<ImplementableArgs>(attr2) {
            Ok(a) => a,
            Err(e) => return e.to_compile_error().into(),
        };
        return implementable_inner(args, trait_item);
    }

    // Parse common args (cfg = "...")
    let export_cfg = match syn::parse2::<ExportCfg>(attr2) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };

    // 2. Enum
    if let Ok(enum_item) = syn::parse2::<DeriveInput>(item2.clone())
        && matches!(enum_item.data, Data::Enum(_))
    {
        return exportable_enum(enum_item, &export_cfg);
    }

    // 3. Free function
    if let Ok(fn_item) = syn::parse2::<syn::ItemFn>(item2) {
        return exportable_free_fn(fn_item, &export_cfg);
    }

    // 4. Impl block — trait impl or inherent impl
    let input = parse_macro_input!(item as ItemImpl);
    if input.trait_.is_some() {
        return trait_impl_inner(input, &export_cfg);
    }
    exportable_struct_impl(input, &export_cfg)
}

/// Common export arguments accepted by all item kinds: `cfg = "predicate"`.
/// Trait definitions accept additional args (reserved, foreign, etc.) via
/// `ImplementableArgs`.
struct ExportCfg {
    /// `#[cfg(...)]` predicate tokens. When set, the proc macro emits the
    /// predicate as `#[cfg(PRED)]` on generated code that references the
    /// gated type, while metadata (for schema/bridge) is emitted unconditionally.
    cfg: Option<proc_macro2::TokenStream>,
}

impl Parse for ExportCfg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut cfg = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            if key == "cfg" {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                cfg = Some(lit.parse::<proc_macro2::TokenStream>()?);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown #[ffier::export] argument `{key}`; expected `cfg`"),
                ));
            }
            let _ = input.parse::<Token![,]>();
        }
        Ok(ExportCfg { cfg })
    }
}

impl ExportCfg {
    /// Emit `#[cfg(PRED)]` token stream if cfg is set.
    fn cfg_attr(&self) -> proc_macro2::TokenStream {
        match &self.cfg {
            Some(pred) => quote! { #[cfg(#pred)] },
            None => quote! {},
        }
    }

    /// Emit `cfg_predicate = (...),` for metadata blobs if cfg is set.
    fn meta_tokens(&self) -> proc_macro2::TokenStream {
        match &self.cfg {
            Some(pred) => quote! { cfg_predicate = (#pred), },
            None => quote! {},
        }
    }
}

fn exportable_struct_impl(input: ItemImpl, export_cfg: &ExportCfg) -> TokenStream {
    // Strip #[ffier(...)] attributes from methods before emitting the impl block
    let impl_block = {
        let mut block = input.clone();
        for item in &mut block.items {
            if let ImplItem::Fn(method) = item {
                strip_ffier_attrs(&mut method.attrs);
                for arg in &mut method.sig.inputs {
                    if let FnArg::Typed(pat_ty) = arg {
                        strip_ffier_attrs(&mut pat_ty.attrs);
                    }
                }
            }
        }
        block
    };

    let Type::Path(ref struct_path) = *input.self_ty else {
        return syn::Error::new_spanned(&input.self_ty, "expected a named struct type")
            .to_compile_error()
            .into();
    };
    let last_seg = struct_path
        .path
        .segments
        .last()
        .expect("expected struct name");
    let struct_ident = &last_seg.ident;
    let self_ty = &input.self_ty;
    let struct_name = struct_ident.to_string();
    let struct_lower = camel_to_snake(&struct_name);

    let helper_mod_name = format_ident!("_ffier_{struct_lower}");
    let mut ctx = AliasContext::new(helper_mod_name.clone());

    let mut methods = Vec::new();
    let is_inherent = input.trait_.is_none();
    let mut warnings = Vec::new();

    for item in &input.items {
        let ImplItem::Fn(method) = item else { continue };

        // Check for forbidden FFI types in params and return
        for arg in &method.sig.inputs {
            if let FnArg::Typed(pat_ty) = arg
                && let Some(msg) = check_forbidden_ffi_type(&pat_ty.ty)
            {
                return syn::Error::new_spanned(&pat_ty.ty, msg)
                    .to_compile_error()
                    .into();
            }
        }
        if let ReturnType::Type(_, ty) = &method.sig.output
            && let Some(msg) = check_forbidden_ffi_type(ty)
        {
            return syn::Error::new_spanned(ty, msg).to_compile_error().into();
        }

        // Skip non-public methods in inherent impls (bridge crate can't call them)
        if is_inherent && !matches!(method.vis, syn::Visibility::Public(_)) {
            let msg = format!(
                "ffier: skipping non-public method `{}`; make it `pub` to export via FFI",
                method.sig.ident
            );
            warnings.push(quote::quote_spanned! { method.sig.ident.span() =>
                const _: () = {
                    #[deprecated = #msg]
                    const WARNING: () = ();
                    let _ = WARNING;
                };
            });
            continue;
        }

        // Parse method-level #[ffier(...)] for foreign_return
        let method_ffier = parse_ffier_method_attrs(&method.attrs).ok();
        if method_ffier
            .as_ref()
            .is_some_and(|a| a.foreign_return.is_some() && a.foreign_return_c_name.is_none())
        {
            return syn::Error::new_spanned(
                &method.sig.ident,
                "#[ffier(foreign_return = ...)] requires c_name = \"...\" \
                 (the C typedef name for the foreign handle type)",
            )
            .to_compile_error()
            .into();
        }
        let foreign_ret = method_ffier
            .as_ref()
            .and_then(|a| a.foreign_return.as_ref());
        let foreign_ret_c = method_ffier
            .as_ref()
            .and_then(|a| a.foreign_return_c_name.as_deref());
        match parse_method_sig(
            &method.sig,
            &method.attrs,
            &mut ctx,
            Some(self_ty),
            false,
            false,
            foreign_ret,
            foreign_ret_c,
        ) {
            Ok(Some(mut m)) => {
                m.ffi_name = format!("{}_{}", struct_lower, method.sig.ident);
                methods.push(m);
            }
            Ok(None) => {}
            Err(e) => return e.to_compile_error().into(),
        }
    }

    // -----------------------------------------------------------------------
    // Metadata emission — structured tokens for generator proc macros
    // -----------------------------------------------------------------------

    let local_type_aliases = ctx.local_type_aliases();

    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_{struct_lower}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{struct_ident}");

    let method_meta_tokens = emit_method_meta(&methods, MethodMetaKind::Impl);

    // Lifetime idents (without the tick) for metadata
    let lifetime_idents: Vec<_> = input
        .generics
        .lifetimes()
        .map(|lt| format_ident!("{}", lt.lifetime.ident))
        .collect();

    let struct_path_tokens = quote! { $crate::#struct_ident };

    // For the @on_library_export arm, we need elided lifetimes so that
    // `impl FfiHandle for View<'_>` compiles in macro_rules! context.
    let elided_lifetimes: Vec<_> = input.generics.lifetimes().map(|_| quote! { '_ }).collect();
    let struct_with_lifetimes = if elided_lifetimes.is_empty() {
        quote! { #struct_ident }
    } else {
        quote! { #struct_ident<#(#elided_lifetimes),*> }
    };

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #impl_block

        #(#warnings)*

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {
            #(#local_type_aliases)*
        }

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            (@on_library_export, $type_tag:expr, [$($__handle:ident),*]) => {
                impl FfiHandle for #struct_with_lifetimes {
                    const C_HANDLE_NAME: &'static str = stringify!(#struct_ident);
                    const TYPE_TAG: u32 = $type_tag;
                }
                impl FfiType for #struct_with_lifetimes {
                    type CRepr = *mut core::ffi::c_void;
                    const C_TYPE_NAME: &'static str = stringify!(#struct_ident);
                    const IS_HANDLE: bool = true;
                    fn into_c(self) -> *mut core::ffi::c_void {
                        ffier::ffier_handle_new($type_tag, self)
                    }
                    unsafe fn from_c(repr: *mut core::ffi::c_void) -> Self {
                        unsafe { ffier::ffier_handle_consume::<Self>(repr) }
                    }
                }
            };
            ($prefix:literal, $type_tag:expr, $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_impl,
                    name = #struct_ident,
                    struct_path = (#struct_path_tokens),
                    prefix = $prefix,
                    type_tag = $type_tag,
                    lifetimes = (#(#lifetime_idents),*),
                    methods = [#(#method_meta_tokens),*],
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

// ---------------------------------------------------------------------------
// #[ffier::export] on enums
// ---------------------------------------------------------------------------

/// Handle `#[ffier::export]` on a `#[repr(uN)]` enum.
///
/// Extracts the repr type and variant discriminants, emits a metadata macro
/// that bridges to the schema generator. Also generates a `FfiType` impl
/// so the enum can be used as a parameter/return type in exported methods.
fn exportable_enum(input: DeriveInput, export_cfg: &ExportCfg) -> TokenStream {
    let name = &input.ident;
    let Data::Enum(data_enum) = &input.data else {
        unreachable!();
    };

    // Extract #[repr(uN)] — required for enum constants.
    let repr_ident = match extract_repr(&input.attrs) {
        Some(r) => r,
        None => {
            return syn::Error::new_spanned(
                &input,
                "#[ffier::export] on enums requires #[repr(u8/u16/u32/u64/i8/i16/i32/i64)]",
            )
            .to_compile_error()
            .into();
        }
    };
    let repr_str = repr_ident.to_string();

    // Collect variant names and discriminant values.
    let mut variants_meta = Vec::new();
    let mut variant_values: Vec<(syn::Ident, u64)> = Vec::new();
    let mut next_value: u64 = 0;
    for variant in &data_enum.variants {
        if !variant.fields.is_empty() {
            return syn::Error::new_spanned(
                variant,
                "#[ffier::export] enums must have unit variants only (no fields)",
            )
            .to_compile_error()
            .into();
        }
        let value = if let Some((_, expr)) = &variant.discriminant {
            // Parse the discriminant expression as a literal integer.
            if let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(lit),
                ..
            }) = expr
            {
                match lit.base10_parse::<u64>() {
                    Ok(v) => v,
                    Err(e) => return e.to_compile_error().into(),
                }
            } else {
                return syn::Error::new_spanned(
                    expr,
                    "ffier: enum discriminant must be a literal integer",
                )
                .to_compile_error()
                .into();
            }
        } else {
            next_value
        };
        next_value = value + 1;
        let var_ident = &variant.ident;
        variant_values.push((var_ident.clone(), value));
        variants_meta.push(quote! {
            { name = #var_ident, value = #value, }
        });
    }

    // Build match arms for from_c: each discriminant maps to its variant,
    // unknown values panic.
    let from_c_arms: Vec<_> = variant_values
        .iter()
        .map(|(ident, val)| {
            let lit = proc_macro2::Literal::u64_unsuffixed(*val);
            quote! { #lit => Self::#ident, }
        })
        .collect();
    let name_str = name.to_string();

    let enum_snake = camel_to_snake(&name.to_string());
    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_{enum_snake}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{name}");
    let helper_mod_name = format_ident!("_ffier_{enum_snake}");

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #input

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {}

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            (@on_library_export, $__type_tag:expr, [$($__handle:ident),*]) => {
                impl FfiType for #name {
                    type CRepr = #repr_ident;
                    const C_TYPE_NAME: &'static str = stringify!(#name);
                    const IS_HANDLE: bool = false;
                    fn into_c(self) -> #repr_ident { self as #repr_ident }
                    unsafe fn from_c(repr: #repr_ident) -> Self {
                        match repr {
                            #(#from_c_arms)*
                            unknown => panic!(
                                "invalid {} discriminant: {}",
                                #name_str, unknown
                            ),
                        }
                    }
                }
                impl FfiBorrow for #name {
                    fn borrow_as_c(&self) -> #repr_ident { *self as #repr_ident }
                }
            };
            ($prefix:literal, $type_tag:expr, $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_enum,
                    name = #name,
                    prefix = $prefix,
                    repr = #repr_str,
                    variants = [#(#variants_meta),*],
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

/// Extract the `#[repr(X)]` attribute from an item, returning the ident X
/// if it's a supported integer repr.
fn extract_repr(attrs: &[syn::Attribute]) -> Option<syn::Ident> {
    for attr in attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        if let Ok(ident) = attr.parse_args::<syn::Ident>() {
            let s = ident.to_string();
            if matches!(
                s.as_str(),
                "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64"
            ) {
                return Some(ident);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// ffier::export_bitflags! — define + export a bitflags type via FFI
// ---------------------------------------------------------------------------

/// Define a `bitflags!` type and export it for FFI.
///
/// Wraps the `bitflags!` macro invocation: the type is always defined,
/// and the FFI metadata (FfiType impl, schema entry) is gated behind
/// the specified Cargo feature.
///
/// ```ignore
/// ffier::export_bitflags! {
///     if = "ffi",
///     bitflags::bitflags! {
///         #[derive(Debug, Clone, Copy, PartialEq, Eq)]
///         pub struct Permissions: u32 {
///             const READ  = 0b001;
///             const WRITE = 0b010;
///             const EXEC  = 0b100;
///         }
///     }
/// }
/// ```
#[proc_macro]
pub fn export_bitflags(input: TokenStream) -> TokenStream {
    let parsed = match syn::parse::<ExportBitflagsInput>(input) {
        Ok(p) => p,
        Err(e) => return e.to_compile_error().into(),
    };

    let bitflags_call = &parsed.bitflags_call;
    let name = &parsed.name;
    let repr_ident = &parsed.repr;
    let repr_str = repr_ident.to_string();
    let export_cfg = ExportCfg {
        cfg: combine_cfg_predicates(&parsed.cfg_predicates),
    };

    let mut variants_meta = Vec::new();
    for (flag_name, value) in &parsed.flags {
        variants_meta.push(quote! {
            { name = #flag_name, value = #value, }
        });
    }

    let bf_snake = camel_to_snake(&name.to_string());
    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_{bf_snake}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{name}");
    let helper_mod_name = format_ident!("_ffier_{bf_snake}");

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #bitflags_call

        // macro_rules! cannot be gated with #[cfg] directly and
        // #[macro_export] cannot appear inside const blocks, so the
        // macro is always emitted but the pub-use alias that makes it
        // reachable is gated behind the feature flag.
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            (@on_library_export, $__type_tag:expr, [$($__handle:ident),*]) => {
                impl FfiType for #name {
                    type CRepr = #repr_ident;
                    const C_TYPE_NAME: &'static str = stringify!(#name);
                    const IS_HANDLE: bool = false;
                    fn into_c(self) -> #repr_ident { self.bits() }
                    unsafe fn from_c(repr: #repr_ident) -> Self { Self::from_bits_retain(repr) }
                }
                impl FfiBorrow for #name {
                    fn borrow_as_c(&self) -> #repr_ident { self.bits() }
                }
            };
            // Tagged invocation (from library_definition! shim): includes prefix
            ($prefix:literal, $type_tag:expr, $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_bitflags,
                    name = #name,
                    prefix = $prefix,
                    repr = #repr_str,
                    variants = [#(#variants_meta),*],
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {}

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

/// Parsed input for `export_bitflags!`:
/// ```ignore
/// bitflags::bitflags! {
///     [attrs]
///     pub struct Name: repr { const FLAG = val; ... }
/// }
/// ```
///
/// An optional `#[cfg(...)]` attribute may precede the `bitflags!` call to
/// gate the type behind a Cargo feature while keeping the FFI metadata
/// unconditionally available:
/// ```ignore
/// ffier::export_bitflags! {
///     #[cfg(feature = "optional")]
///     bitflags::bitflags! { ... }
/// }
/// ```
struct ExportBitflagsInput {
    /// `#[cfg(...)]` predicates extracted from before the `bitflags!` call.
    cfg_predicates: Vec<syn::Meta>,
    /// The complete `bitflags! { ... }` or `bitflags::bitflags! { ... }`
    /// invocation, preserved verbatim for re-emission.
    bitflags_call: proc_macro2::TokenStream,
    /// Struct name extracted from the inner body.
    name: syn::Ident,
    /// Repr type (u8/u16/u32/u64/i8/i16/i32/i64).
    repr: syn::Ident,
    /// Flag constants: (name, value).
    flags: Vec<(syn::Ident, u64)>,
}

impl Parse for ExportBitflagsInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Parse optional leading #[cfg(...)] attributes.
        let mut cfg_predicates = Vec::new();
        while input.peek(Token![#]) && input.peek2(syn::token::Bracket) {
            let attrs = input.call(syn::Attribute::parse_outer)?;
            for attr in attrs {
                if attr.path().is_ident("cfg") {
                    cfg_predicates.push(attr.parse_args::<syn::Meta>()?);
                } else {
                    return Err(syn::Error::new_spanned(
                        attr,
                        "export_bitflags! only supports #[cfg(...)] attributes before the bitflags! call",
                    ));
                }
            }
        }

        // Capture the remaining tokens as the bitflags! call.
        // We need to parse them to extract name/repr/flags, but also
        // preserve the original tokens for verbatim re-emission.
        let remaining: proc_macro2::TokenStream = input.parse()?;
        let bitflags_call = remaining.clone();

        // Parse the bitflags macro call to extract struct info.
        // Expected forms:
        //   bitflags! { ... }
        //   bitflags::bitflags! { ... }
        //   path::to::bitflags! { ... }
        let body = parse_bitflags_body(remaining)?;

        Ok(ExportBitflagsInput {
            cfg_predicates,
            bitflags_call,
            name: body.name,
            repr: body.repr,
            flags: body.flags,
        })
    }
}

/// Extract struct name, repr, and flag constants from a `bitflags! { ... }` call.
///
/// Parses through the macro path to find the braced body, then scans for
/// `struct Name: repr { const FLAG = val; ... }` inside it, skipping any
/// leading attributes.
fn parse_bitflags_body(tokens: proc_macro2::TokenStream) -> syn::Result<BitflagsStructBody> {
    // Find the `!` and the braced body after the macro path.
    let mut iter = tokens.into_iter().peekable();
    let mut found_bang = false;

    for tt in iter.by_ref() {
        if let proc_macro2::TokenTree::Punct(p) = &tt
            && p.as_char() == '!'
        {
            found_bang = true;
            break;
        }
    }

    if !found_bang {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "expected `bitflags! { ... }` macro invocation",
        ));
    }

    // The next token should be a brace-delimited group
    let body_group = match iter.next() {
        Some(proc_macro2::TokenTree::Group(g))
            if g.delimiter() == proc_macro2::Delimiter::Brace =>
        {
            g
        }
        _ => {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "expected braced body after `bitflags!`",
            ));
        }
    };

    // Parse the inner body to find `struct Name: repr { ... }`
    // Skip leading attributes (#[...]) and visibility (pub)
    let body_stream = body_group.stream();
    syn::parse2::<BitflagsStructBody>(body_stream)
}

/// Parser for the inside of a `bitflags! { ... }` body.
/// Skips attributes, parses `[pub] struct Name: repr { const FLAG = val; ... }`.
struct BitflagsStructBody {
    name: syn::Ident,
    repr: syn::Ident,
    flags: Vec<(syn::Ident, u64)>,
}

impl Parse for BitflagsStructBody {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Skip outer attributes (#[derive(...)], #[repr(...)], etc.)
        let _ = input.call(syn::Attribute::parse_outer)?;

        // Skip optional visibility
        let _: Option<syn::Visibility> = if input.peek(Token![pub]) {
            Some(input.parse()?)
        } else {
            None
        };

        // `struct`
        input.parse::<Token![struct]>()?;

        // Name
        let name: syn::Ident = input.parse()?;

        // `: repr`
        input.parse::<Token![:]>()?;
        let repr: syn::Ident = input.parse()?;

        // Validate repr
        let repr_s = repr.to_string();
        if !matches!(
            repr_s.as_str(),
            "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64"
        ) {
            return Err(syn::Error::new(
                repr.span(),
                "bitflags repr must be one of u8, u16, u32, u64, i8, i16, i32, i64",
            ));
        }

        // `{ const FLAG = val; ... }`
        let content;
        syn::braced!(content in input);

        let mut flags = Vec::new();
        while !content.is_empty() {
            // Skip attributes on individual flags (e.g. #[doc = "..."])
            let _ = content.call(syn::Attribute::parse_outer)?;

            if content.is_empty() {
                break;
            }

            content.parse::<Token![const]>()?;
            let flag_name: syn::Ident = content.parse()?;
            content.parse::<Token![=]>()?;

            let lit: syn::LitInt = content.parse()?;
            let value = lit.base10_parse::<u64>()?;

            flags.push((flag_name, value));

            // Optional semicolon and/or comma
            let _ = content.parse::<Token![;]>();
            let _ = content.parse::<Token![,]>();
        }

        Ok(BitflagsStructBody { name, repr, flags })
    }
}

// ---------------------------------------------------------------------------
// #[ffier::export] on free functions
// ---------------------------------------------------------------------------

/// Handle `#[ffier::export]` on a free (non-method) function.
fn exportable_free_fn(input: syn::ItemFn, export_cfg: &ExportCfg) -> TokenStream {
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();

    let helper_mod_name = format_ident!("_ffier_fn_{fn_name_str}");
    let mut ctx = AliasContext::new(helper_mod_name.clone());

    // Parse the function signature as a static method (no self)
    let method_ffier = parse_ffier_method_attrs(&input.attrs).ok();
    if method_ffier
        .as_ref()
        .is_some_and(|a| a.foreign_return.is_some() && a.foreign_return_c_name.is_none())
    {
        return syn::Error::new_spanned(
            &input.sig.ident,
            "#[ffier(foreign_return = ...)] requires c_name = \"...\" \
             (the C typedef name for the foreign handle type)",
        )
        .to_compile_error()
        .into();
    }
    let foreign_ret = method_ffier
        .as_ref()
        .and_then(|a| a.foreign_return.as_ref());
    let foreign_ret_c = method_ffier
        .as_ref()
        .and_then(|a| a.foreign_return_c_name.as_deref());
    let mut method = match parse_method_sig(
        &input.sig,
        &input.attrs,
        &mut ctx,
        None,
        false,
        false,
        foreign_ret,
        foreign_ret_c,
    ) {
        Ok(Some(m)) => m,
        Err(e) => return e.to_compile_error().into(),
        Ok(None) => {
            return syn::Error::new_spanned(
                &input.sig,
                "ffier: could not parse free function signature",
            )
            .to_compile_error()
            .into();
        }
    };
    method.ffi_name = fn_name_str.clone();

    let local_type_aliases = ctx.local_type_aliases();
    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_fn_{fn_name_str}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{fn_name}");

    let method_meta_tokens = emit_method_meta(&[method], MethodMetaKind::Impl);
    let fn_path = quote! { $crate::#fn_name };
    let doc_lines = extract_doc_comments(&input.attrs);

    // Strip #[ffier(...)] attributes from the function and its params
    let clean_fn = {
        let mut f = input.clone();
        strip_ffier_attrs(&mut f.attrs);
        for arg in &mut f.sig.inputs {
            if let FnArg::Typed(pat_ty) = arg {
                strip_ffier_attrs(&mut pat_ty.attrs);
            }
        }
        f
    };

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #clean_fn

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {
            #(#local_type_aliases)*
        }

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            ($prefix:literal, $type_tag:expr, $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_fn,
                    name = #fn_name,
                    fn_path = (#fn_path),
                    prefix = $prefix,
                    ffi_name = #fn_name_str,
                    doc = [#(#doc_lines),*],
                    methods = [#(#method_meta_tokens),*],
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_result_types(ty: &Type) -> Option<(Type, Type)> {
    let Type::Path(tp) = ty else { return None };
    let last = tp.path.segments.last()?;
    if last.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &last.arguments else {
        return None;
    };
    let mut iter = args.args.iter();
    let GenericArgument::Type(ok_ty) = iter.next()? else {
        return None;
    };
    let GenericArgument::Type(err_ty) = iter.next()? else {
        return None;
    };
    Some((ok_ty.clone(), err_ty.clone()))
}

fn type_ident_name(ty: &Type) -> String {
    let Type::Path(tp) = ty else {
        return "Error".to_string();
    };
    tp.path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_else(|| "Error".to_string())
}

fn is_unit_type(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(t) if t.elems.is_empty())
}

/// Check if `ty` is the same type as `target` or a reference to it.
/// Matches: `Self`, `&Self`, `&mut Self` (after Self→concrete replacement).
fn is_self_return(ty: &Type, target: &Type) -> bool {
    let tgt = target.to_token_stream().to_string().replace(' ', "");
    let inner = match ty {
        Type::Reference(ref_ty) => ref_ty.elem.to_token_stream().to_string().replace(' ', ""),
        _ => ty.to_token_stream().to_string().replace(' ', ""),
    };
    inner == tgt
}

/// Replace all occurrences of `Self` in a type with a concrete type.
fn replace_self_type(ty: &Type, replacement: &Type) -> Type {
    struct Replacer<'a>(&'a Type);
    impl VisitMut for Replacer<'_> {
        fn visit_type_mut(&mut self, ty: &mut Type) {
            if let Type::Path(tp) = ty
                && tp.path.is_ident("Self")
            {
                *ty = self.0.clone();
                return;
            }
            syn::visit_mut::visit_type_mut(self, ty);
        }
    }
    let mut ty = ty.clone();
    Replacer(replacement).visit_type_mut(&mut ty);
    ty
}

/// Tracks type aliases needed for cross-crate `$crate::` resolution in metadata macros.
struct AliasContext {
    types: Vec<Type>,
    aliases: Vec<syn::Ident>,
    counter: u32,
    helper_mod: syn::Ident,
}

/// Check if a TypePath is `Option<_>`.
fn is_option(tp: &syn::TypePath) -> bool {
    tp.qself.is_none() && {
        let last = tp.path.segments.last().unwrap();
        last.ident == "Option" && matches!(&last.arguments, syn::PathArguments::AngleBracketed(_))
    }
}

/// Extract the inner type from `Option<T>`.
fn option_inner(tp: &syn::TypePath) -> &Type {
    let last = tp.path.segments.last().unwrap();
    match &last.arguments {
        syn::PathArguments::AngleBracketed(args) => match args.args.first().unwrap() {
            syn::GenericArgument::Type(ty) => ty,
            _ => panic!("Option<_> must have a type argument"),
        },
        _ => panic!("Option must have angle bracket arguments"),
    }
}

/// Check if a TypePath is `Box<str>`.
fn is_box_str(tp: &syn::TypePath) -> bool {
    tp.qself.is_none() && {
        let last = tp.path.segments.last().unwrap();
        if last.ident != "Box" {
            return false;
        }
        match &last.arguments {
            syn::PathArguments::AngleBracketed(args) => match args.args.first() {
                Some(syn::GenericArgument::Type(Type::Path(inner))) => inner.path.is_ident("str"),
                _ => false,
            },
            _ => false,
        }
    }
}

const PRIMITIVES: &[&str] = &[
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "isize", "usize", "bool", "f32", "f64",
];

impl AliasContext {
    fn new(helper_mod: syn::Ident) -> Self {
        Self {
            types: Vec::new(),
            aliases: Vec::new(),
            counter: 0,
            helper_mod,
        }
    }

    /// Produce bridge_type tokens for any Rust type, including references.
    ///
    /// Recursively handles reference and slice types, producing tokens
    /// that resolve via `<T as FfiType>::CRepr` in the cdylib context.
    /// Primitives and `str` are emitted directly; everything else gets
    /// a `$crate::helper_mod::_TypeN` alias for cross-crate resolution.
    fn bridge_tokens(&mut self, ty: &Type) -> proc_macro2::TokenStream {
        match ty {
            Type::Reference(ref_ty) => {
                let inner = self.bridge_tokens(&ref_ty.elem);
                if ref_ty.mutability.is_some() {
                    quote! { &'static mut #inner }
                } else {
                    quote! { &'static #inner }
                }
            }
            // Raw pointers (*mut T, *const T) — recurse into the pointee.
            // The user must write fully qualified paths (e.g. `*mut core::ffi::c_void`).
            Type::Ptr(ptr_ty) => {
                let inner = self.bridge_tokens(&ptr_ty.elem);
                if ptr_ty.mutability.is_some() {
                    quote! { *mut #inner }
                } else {
                    quote! { *const #inner }
                }
            }
            Type::Slice(sl) => {
                let elem = self.bridge_tokens(&sl.elem);
                quote! { [#elem] }
            }
            Type::Path(tp) if tp.path.is_ident("str") => quote! { str },
            Type::Path(tp) if is_option(tp) => {
                let inner_ty = option_inner(tp);
                let inner = self.bridge_tokens(inner_ty);
                quote! { Option<#inner> }
            }
            Type::Path(tp) if is_box_str(tp) => quote! { Box<str> },
            _ => self.alias_tokens(ty),
        }
    }

    /// Get or create an alias for a non-reference, non-slice, non-keyword type.
    fn alias_tokens(&mut self, ty: &Type) -> proc_macro2::TokenStream {
        if is_primitive(ty) {
            return quote! { #ty };
        }
        for (i, existing) in self.types.iter().enumerate() {
            if existing == ty {
                let alias = &self.aliases[i];
                let helper = &self.helper_mod;
                return quote! { $crate::#helper::#alias };
            }
        }
        let alias = format_ident!("_Type{}", self.counter);
        self.counter += 1;
        self.types.push(ty.clone());
        self.aliases.push(alias.clone());
        let helper = &self.helper_mod;
        quote! { $crate::#helper::#alias }
    }

    /// Emit `pub type _TypeN = super::Erased;` items for the helper module
    /// emitted at the definition site.
    ///
    /// Types that start with an external crate path (multi-segment, not
    /// starting with `crate` or `self` or `super`) are emitted without
    /// `super::` since they're already absolute paths.
    fn local_type_aliases(&self) -> Vec<proc_macro2::TokenStream> {
        self.types
            .iter()
            .zip(self.aliases.iter())
            .map(|(ty, alias)| {
                let erased = erase_lifetimes(ty);
                if is_external_crate_path(&erased) {
                    quote! { pub type #alias = #erased; }
                } else {
                    quote! { pub type #alias = super::#erased; }
                }
            })
            .collect()
    }
}

fn is_primitive(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    tp.path.segments.len() == 1
        && PRIMITIVES.contains(&tp.path.segments[0].ident.to_string().as_str())
}

/// Check if a type starts with an external crate path (e.g. `other_crate::Foo`).
/// Such types should NOT get `super::` prepended in helper module aliases.
fn is_external_crate_path(ty: &Type) -> bool {
    let Type::Path(tp) = ty else { return false };
    if tp.path.segments.len() <= 1 {
        return false;
    }
    let first = tp.path.segments.first().unwrap().ident.to_string();
    // `crate::`, `self::`, `super::` are local paths, not external
    !matches!(first.as_str(), "crate" | "self" | "super")
}

// ---------------------------------------------------------------------------
// #[derive(FfiError)]
// ---------------------------------------------------------------------------

/// Derive `ffier::FfiError` for a unit enum.
///
/// Each variant must have `#[ffier(code = N)]` and optionally
/// `#[ffier(code = N, message = "...")]`. Without an explicit message,
/// the variant name is humanized (`DivisionByZero` → `"division by zero"`).
///
/// ```ignore
/// #[derive(ffier::FfiError)]
/// pub enum CalcError {
///     #[ffier(code = 1)]
///     DivisionByZero,
///     #[ffier(code = 2, message = "integer overflow")]
///     Overflow,
/// }
/// ```
#[proc_macro_derive(FfiError, attributes(ffier))]
pub fn derive_ffi_error(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let Data::Enum(data_enum) = &input.data else {
        return syn::Error::new_spanned(&input, "FfiError can only be derived for enums")
            .to_compile_error()
            .into();
    };

    // Validate: enum must be #[non_exhaustive]
    if !input
        .attrs
        .iter()
        .any(|a| a.path().is_ident("non_exhaustive"))
    {
        return syn::Error::new_spanned(&input, "FfiError enums must be #[non_exhaustive]")
            .to_compile_error()
            .into();
    }

    // Validate: no unit variants (use `Variant()` instead of `Variant`)
    for variant in &data_enum.variants {
        if matches!(variant.fields, syn::Fields::Unit) {
            return syn::Error::new_spanned(
                variant,
                "FfiError variants must not be unit variants; \
                 use `Variant()` instead of `Variant` for FFI compatibility",
            )
            .to_compile_error()
            .into();
        }
        if matches!(variant.fields, syn::Fields::Named(_)) {
            return syn::Error::new_spanned(
                variant,
                "FfiError: named fields in variants are not yet supported; \
                 use tuple variants like `Variant(Box<str>)`",
            )
            .to_compile_error()
            .into();
        }
    }

    let mut code_arms = Vec::new();
    let mut message_arms = Vec::new();
    let mut codes_entries = Vec::new();
    let mut variant_meta_tokens = Vec::new();
    let mut payload_arms = Vec::new();

    for variant in &data_enum.variants {
        let var_ident = &variant.ident;

        let attrs = match parse_ffier_variant_attrs(&variant.attrs) {
            Ok(a) => a,
            Err(e) => return e.to_compile_error().into(),
        };

        let code = attrs.code;
        // Static message for strerror: #[ffier(message="...")] > raw variant name
        // Data-carrying variants get "Name(...)" / "Name{..}" to signal
        // that ft_error_message() has richer detail.
        let message = attrs.message.unwrap_or_else(|| {
            let name = var_ident.to_string();
            match &variant.fields {
                syn::Fields::Unit => name,
                syn::Fields::Unnamed(f) if f.unnamed.is_empty() => name,
                syn::Fields::Unnamed(_) => format!("{name}(...)"),
                syn::Fields::Named(_) => format!("{name}{{..}}"),
            }
        });
        let upper_name = camel_to_upper_snake(&var_ident.to_string());

        // Wildcard pattern for variants with fields
        let match_pattern = match &variant.fields {
            syn::Fields::Unit => quote! { #name::#var_ident },
            syn::Fields::Unnamed(_) => quote! { #name::#var_ident(..) },
            syn::Fields::Named(_) => quote! { #name::#var_ident { .. } },
        };
        code_arms.push(quote! { #match_pattern => #code });

        let msg_with_nul = format!("{message}\0");
        let msg_lit = proc_macro2::Literal::byte_string(msg_with_nul.as_bytes());
        message_arms.push(quote! {
            #code => unsafe {
                core::ffi::CStr::from_bytes_with_nul_unchecked(#msg_lit)
            }
        });

        // Collect field types for data-carrying variants.
        // Each type is emitted as a parenthesized token group `(Type)` so the
        // meta crate can parse it back as a TokenStream without stringifying.
        // Opaque variants have no marshallable fields: exclude them from metadata.
        let field_types: Vec<_> = if attrs.opaque {
            vec![]
        } else {
            match &variant.fields {
                syn::Fields::Unit => vec![],
                syn::Fields::Unnamed(fields) => fields
                    .unnamed
                    .iter()
                    .map(|f| {
                        let ty = &f.ty;
                        quote! { (#ty) }
                    })
                    .collect(),
                syn::Fields::Named(_) => {
                    return syn::Error::new_spanned(
                        variant,
                        "FfiError: named fields in variants are not yet supported; \
                         use tuple variants like Variant(Box<str>)",
                    )
                    .to_compile_error()
                    .into();
                }
            }
        };

        // Build payload arm for this variant.
        // Opaque variants suppress marshalling — the inner value is
        // Rust-only and not transferable across the C boundary.
        if attrs.opaque {
            payload_arms.push(quote! {
                #name::#var_ident(..) => {}
            });
        } else {
            match &variant.fields {
                syn::Fields::Unnamed(fields) if fields.unnamed.is_empty() => {
                    // Empty-tuple variant — no payload to write
                    payload_arms.push(quote! {
                        #name::#var_ident(..) => {}
                    });
                }
                syn::Fields::Unnamed(fields) if fields.unnamed.len() == 1 => {
                    let field_ty = &fields.unnamed[0].ty;
                    payload_arms.push(quote! {
                        #name::#var_ident(val, ..) => {
                            let expected = core::mem::size_of::<<#field_ty as FfiType>::CRepr>();
                            assert!(
                                buf_size >= expected,
                                "error payload buffer too small: got {} bytes, need {}",
                                buf_size, expected,
                            );
                            let c_val = <#field_ty as FfiBorrow>::borrow_as_c(val);
                            unsafe {
                                core::ptr::write(
                                    out_buf as *mut <#field_ty as FfiType>::CRepr,
                                    c_val,
                                );
                            }
                        }
                    });
                }
                _ => {} // already rejected above
            }
        }

        codes_entries.push(quote! { (#upper_name, #code) });
        variant_meta_tokens.push(quote! {
            { name = #var_ident, code = #code, message = #message, fields = [#(#field_types),*], }
        });
    }

    let unknown_msg = format!(
        "unknown {} error\0",
        camel_to_snake(&name.to_string()).replace('_', " ")
    );
    let unknown_lit = proc_macro2::Literal::byte_string(unknown_msg.as_bytes());

    let error_snake = camel_to_snake(&name.to_string());
    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_{error_snake}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{name}");

    let error_path = quote! { $crate::#name };
    let helper_mod_name = format_ident!("_ffier_{error_snake}");

    let output = quote! {
        #[doc(hidden)]
        pub mod #helper_mod_name {}

        impl ffier::FfiError for #name {
            fn code(&self) -> u32 {
                match self {
                    #(#code_arms,)*
                }
            }

            fn static_message(code: u32) -> &'static core::ffi::CStr {
                match code {
                    #(#message_arms,)*
                    _ => unsafe {
                        core::ffi::CStr::from_bytes_with_nul_unchecked(#unknown_lit)
                    },
                }
            }

            fn codes() -> &'static [(&'static str, u32)] {
                &[#(#codes_entries),*]
            }

            unsafe fn payload(&self, out_buf: *mut core::ffi::c_void, buf_size: usize) {
                match self {
                    #(#payload_arms)*
                    #[allow(unreachable_patterns)]
                    _ => {}
                }
            }
        }

        #[ffier::export]
        impl ffier::Error for #name {
            fn code(&self) -> u32 {
                ffier::FfiError::code(self)
            }
            fn message(&self, writer: &mut impl ffier::PushStr) {
                use core::fmt::Write;
                let writer: &mut dyn ffier::PushStr = writer;
                let _ = write!(writer, "{}", self);
            }
        }

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            (@on_library_export, $type_tag:expr, [$($__handle:ident),*]) => {
                impl FfiHandle for #name {
                    const C_HANDLE_NAME: &'static str = stringify!(#name);
                    const TYPE_TAG: u32 = $type_tag;
                }
                impl FfiType for #name {
                    type CRepr = *mut core::ffi::c_void;
                    const C_TYPE_NAME: &'static str = stringify!(#name);
                    const IS_HANDLE: bool = true;
                    fn into_c(self) -> *mut core::ffi::c_void {
                        ffier::ffier_handle_new($type_tag, self)
                    }
                    unsafe fn from_c(repr: *mut core::ffi::c_void) -> Self {
                        unsafe { ffier::ffier_handle_consume::<Self>(repr) }
                    }
                }
            };
            // Tagged invocation (from library_definition! shim): includes type_tag
            ($prefix:literal, $type_tag:expr, $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_error,
                    name = #name,
                    path = (#error_path),
                    prefix = $prefix,
                    type_tag = $type_tag,
                    variants = [#(#variant_meta_tokens),*],
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

struct FfierVariantAttrs {
    code: u32,
    message: Option<String>,
    /// When true, the variant's field (if any) is not marshalled across FFI.
    /// The payload arm becomes a no-op and the field is excluded from schema
    /// metadata. Use this for fields whose types don't implement `FfiType`
    /// (e.g. `anyhow::Error`).
    opaque: bool,
}

fn parse_ffier_variant_attrs(attrs: &[syn::Attribute]) -> syn::Result<FfierVariantAttrs> {
    for raw_attr in attrs {
        let Some(attr) = extract_ffier_attr(raw_attr) else {
            continue;
        };
        let attr = &attr;

        let mut code = None;
        let mut message = None;
        let mut opaque = false;

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("code") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                code = Some(lit.base10_parse::<u32>()?);
                Ok(())
            } else if meta.path.is_ident("message") {
                let value = meta.value()?;
                let lit: LitStr = value.parse()?;
                message = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("opaque") {
                opaque = true;
                Ok(())
            } else {
                Err(meta.error("expected `code`, `message`, or `opaque`"))
            }
        })?;

        let code = code
            .ok_or_else(|| syn::Error::new_spanned(attr, "missing `code` in #[ffier(code = N)]"))?;

        return Ok(FfierVariantAttrs {
            code,
            message,
            opaque,
        });
    }

    Err(syn::Error::new(
        proc_macro2::Span::call_site(),
        "missing #[ffier(code = N)] attribute on variant",
    ))
}

/// Parse `#[ffier(dispatch = concrete|vtable)]` from a parameter's attributes.
/// Only `dispatch` is recognized; unknown keys are rejected.
/// Recognized `#[ffier(...)]` attributes on a parameter.
struct FfierParamAttrs {
    dispatch: Option<String>,
    /// Foreign ffier library crate path (e.g. `other_lib`).
    foreign_crate: Option<syn::Path>,
    /// C typedef name for the foreign handle type (e.g. `"FlForeignConfig"`).
    foreign_c_name: Option<String>,
}

fn parse_ffier_param_attrs(attrs: &[syn::Attribute]) -> syn::Result<FfierParamAttrs> {
    let mut result = FfierParamAttrs {
        dispatch: None,
        foreign_crate: None,
        foreign_c_name: None,
    };
    for raw_attr in attrs {
        let Some(attr) = extract_ffier_attr(raw_attr) else {
            continue;
        };
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("dispatch") {
                let value = meta.value()?;
                let mode: syn::Ident = value.parse()?;
                result.dispatch = Some(mode.to_string());
            } else if meta.path.is_ident("foreign") {
                let value = meta.value()?;
                let path: syn::Path = value.parse()?;
                result.foreign_crate = Some(path);
            } else if meta.path.is_ident("c_name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                result.foreign_c_name = Some(lit.value());
            } else {
                return Err(meta.error(format!(
                    "unknown #[ffier] key `{}` on parameter",
                    meta.path.to_token_stream(),
                )));
            }
            Ok(())
        })?;
    }
    Ok(result)
}

/// All recognized keys from `#[ffier(...)]` on a trait/impl method.
struct FfierMethodAttrs {
    index: Option<usize>,
    raw_handle: bool,
    dispatch: Option<String>,
    skip: bool,
    /// Foreign ffier library crate path for the return type.
    foreign_return: Option<syn::Path>,
    /// C typedef name for the foreign return handle type.
    foreign_return_c_name: Option<String>,
}

/// Parse all `#[ffier(...)]` attributes on a method in one pass.
/// Rejects unknown keys.
fn parse_ffier_method_attrs(attrs: &[syn::Attribute]) -> syn::Result<FfierMethodAttrs> {
    let mut result = FfierMethodAttrs {
        index: None,
        raw_handle: false,
        dispatch: None,
        skip: false,
        foreign_return: None,
        foreign_return_c_name: None,
    };

    for raw_attr in attrs {
        let Some(attr) = extract_ffier_attr(raw_attr) else {
            continue;
        };
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("index") {
                let value = meta.value()?;
                let lit: syn::LitInt = value.parse()?;
                result.index = Some(lit.base10_parse::<usize>()?);
            } else if meta.path.is_ident("raw_handle") {
                result.raw_handle = true;
            } else if meta.path.is_ident("dispatch") {
                let value = meta.value()?;
                let mode: syn::Ident = value.parse()?;
                result.dispatch = Some(mode.to_string());
            } else if meta.path.is_ident("skip") {
                result.skip = true;
            } else if meta.path.is_ident("foreign_return") {
                let value = meta.value()?;
                let path: syn::Path = value.parse()?;
                result.foreign_return = Some(path);
            } else if meta.path.is_ident("c_name") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                result.foreign_return_c_name = Some(lit.value());
            } else {
                return Err(meta.error(format!(
                    "unknown #[ffier] key `{}`",
                    meta.path.to_token_stream(),
                )));
            }
            Ok(())
        })?;
    }

    Ok(result)
}

/// Extract the trait name from an `impl Trait` type.
/// Extract trait name and lifetime arguments from `impl Trait<'a, 'b>`.
/// Returns `(trait_name, lifetime_args)` e.g. `("Snapshot", vec!["a"])`.
fn extract_impl_trait_info(ty: &Type) -> Option<(String, Vec<String>)> {
    if let Type::ImplTrait(impl_trait) = ty {
        for bound in &impl_trait.bounds {
            if let syn::TypeParamBound::Trait(trait_bound) = bound
                && let Some(seg) = trait_bound.path.segments.last()
            {
                let name = seg.ident.to_string();
                let mut lt_args = Vec::new();
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Lifetime(lt) = arg {
                            lt_args.push(lt.ident.to_string());
                        }
                    }
                }
                return Some((name, lt_args));
            }
        }
    }
    None
}

/// Extract `#[cfg(...)]` predicates from a list of attributes.
/// Only matches direct `#[cfg(...)]` attributes — does not unwrap
/// `#[cfg_attr(PRED, ...)]`.
/// Returns the predicates (e.g. `feature = "x"`) as `syn::Meta` values.
fn extract_cfg_predicates(attrs: &[syn::Attribute]) -> Vec<syn::Meta> {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("cfg") {
                attr.parse_args::<syn::Meta>().ok()
            } else {
                None
            }
        })
        .collect()
}

/// Combine cfg predicates into a single token stream.
/// Multiple predicates are combined with `all(...)`.
fn combine_cfg_predicates(predicates: &[syn::Meta]) -> Option<proc_macro2::TokenStream> {
    match predicates {
        [] => None,
        [p] => Some(p.to_token_stream()),
        preds => Some(quote! { all(#(#preds),*) }),
    }
}

fn extract_doc_comments(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            let syn::Meta::NameValue(nv) = &attr.meta else {
                return None;
            };
            let syn::Expr::Lit(lit) = &nv.value else {
                return None;
            };
            let syn::Lit::Str(s) = &lit.lit else {
                return None;
            };
            Some(s.value())
        })
        .collect()
}

// ===========================================================================
// #[ffier::export] on trait definitions — C users can implement a Rust trait via vtable
// ===========================================================================

struct ImplementableArgs {
    /// Reserved vtable slot indices (retired methods). These slots are padded
    /// in the vtable struct to keep the layout stable.
    reserved: Vec<usize>,
    /// If true, the trait is foreign (defined in another crate). The macro
    /// will not emit the trait definition or `&mut dyn Trait` dispatch impl.
    foreign: bool,
    /// Optional blessing tag for well-known types (e.g. `"error_trait"`).
    bless: Option<String>,
    /// If true, no vtable wrapper type is generated. C callers cannot
    /// implement this trait — only concrete Rust implementors are dispatched.
    /// Useful for marker traits and traits with supertrait bounds.
    no_vtable: bool,
    /// `#[cfg(...)]` predicate for conditional compilation.
    export_cfg: ExportCfg,
}

impl Parse for ImplementableArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut reserved = Vec::new();
        let mut foreign = false;
        let mut bless = None;
        let mut no_vtable = false;
        let mut cfg = None;

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;

            if ident == "prefix" {
                input.parse::<Token![=]>()?;
                let _lit: LitStr = input.parse()?;
            } else if ident == "reserved" {
                let content;
                syn::parenthesized!(content in input);
                while !content.is_empty() {
                    let lit: syn::LitInt = content.parse()?;
                    reserved.push(lit.base10_parse::<usize>()?);
                    let _ = content.parse::<Token![,]>();
                }
            } else if ident == "foreign" {
                foreign = true;
            } else if ident == "bless" {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                bless = Some(lit.value());
            } else if ident == "no_vtable" {
                no_vtable = true;
            } else if ident == "cfg" {
                input.parse::<Token![=]>()?;
                let lit: LitStr = input.parse()?;
                cfg = Some(lit.parse::<proc_macro2::TokenStream>()?);
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected `prefix`, `reserved`, `foreign`, `bless`, `no_vtable`, or `cfg`",
                ));
            }
            let _ = input.parse::<Token![,]>();
        }

        Ok(Self {
            reserved,
            foreign,
            bless,
            no_vtable,
            export_cfg: ExportCfg { cfg },
        })
    }
}

// ---------------------------------------------------------------------------
// Unified method parsing — shared by all #[ffier::export] item kinds
// ---------------------------------------------------------------------------

fn extract_vtable_methods(
    trait_item: &ItemTrait,
    reserved: &[usize],
    ctx: &mut AliasContext,
) -> syn::Result<Vec<MethodInfo>> {
    let mut methods = Vec::new();

    for item in &trait_item.items {
        let TraitItem::Fn(method) = item else {
            continue;
        };
        let has_default = method.default.is_some();
        let mattrs = parse_ffier_method_attrs(&method.attrs)?;
        if let Some(mut m) = parse_method_sig(
            &method.sig,
            &method.attrs,
            ctx,
            None,
            has_default,
            mattrs.raw_handle,
            mattrs.foreign_return.as_ref(),
            mattrs.foreign_return_c_name.as_deref(),
        )? {
            let index = mattrs.index.ok_or_else(|| {
                syn::Error::new_spanned(
                    &method.sig.ident,
                    format!(
                        "vtable method `{}` is missing `#[ffier(index = N)]`",
                        method.sig.ident,
                    ),
                )
            })?;
            m.index = index;
            methods.push(m);
        }
    }

    // Validate: no duplicate indices
    let mut seen: HashMap<usize, &syn::Ident> = HashMap::new();
    for m in &methods {
        if let Some(prev) = seen.insert(m.index, &m.name) {
            return Err(syn::Error::new_spanned(
                &m.name,
                format!(
                    "duplicate vtable index {}: both `{}` and `{}` use index {}",
                    m.index, prev, m.name, m.index,
                ),
            ));
        }
    }

    // Validate: no method uses a reserved index
    for m in &methods {
        if reserved.contains(&m.index) {
            return Err(syn::Error::new_spanned(
                &m.name,
                format!(
                    "vtable index {} is reserved (retired slot) but used by method `{}`",
                    m.index, m.name,
                ),
            ));
        }
    }

    Ok(methods)
}

/// Parse a single method signature into a unified `MethodInfo`.
///
/// - `attrs`: method-level attributes (for doc comments and `#[ffier(dispatch = ...)]` on params)
/// - `self_ty`: if `Some`, `Self` in param/return types is replaced with this type
///   and builder pattern is detected. Typically `Some` for exported struct impls, `None`
///   for trait definitions and trait impls.
/// - `has_default`: whether this method has a default impl body (trait methods only)
/// - `raw_handle`: whether this is a raw-handle method
#[allow(clippy::too_many_arguments)]
fn parse_method_sig(
    sig: &syn::Signature,
    attrs: &[syn::Attribute],
    ctx: &mut AliasContext,
    self_ty: Option<&Type>,
    has_default: bool,
    raw_handle: bool,
    foreign_return: Option<&syn::Path>,
    foreign_return_c_name: Option<&str>,
) -> syn::Result<Option<MethodInfo>> {
    // --- Determine receiver ---
    let receiver = if raw_handle {
        Receiver::None
    } else {
        match sig.inputs.first() {
            Some(FnArg::Receiver(r)) => {
                if r.reference.is_none() {
                    Receiver::Value
                } else if r.mutability.is_some() {
                    Receiver::Mut
                } else {
                    Receiver::Ref
                }
            }
            _ if self_ty.is_some() => Receiver::None, // static method in exported impl
            _ if self_ty.is_none() && !has_default => Receiver::None, // free function
            _ => return Ok(None),                     // trait method without receiver — skip
        }
    };

    // Skip receiver or raw_handle's first param (the handle pointer)
    let skip_n = if receiver != Receiver::None || raw_handle {
        1
    } else {
        0
    };

    // --- Parse params ---
    let params: Vec<_> = sig
        .inputs
        .iter()
        .skip(skip_n)
        .filter_map(|arg| {
            let FnArg::Typed(pt) = arg else { return None };
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            Some((pi, pt))
        })
        .map(|(pi, pt)| -> syn::Result<ParamInfo> {
            // Unwrap reference for impl Trait detection:
            // `&mut impl PushStr` → inner type is `impl PushStr`, ref_kind = "mut"
            let (inner_ty, impl_trait_ref_kind) = match &*pt.ty {
                Type::Reference(r) => {
                    let rk = if r.mutability.is_some() { "mut" } else { "ref" };
                    (&*r.elem, rk)
                }
                other => (other, "value"),
            };

            // Parse #[ffier(...)] attrs on the parameter
            let param_attrs = parse_ffier_param_attrs(&pt.attrs)?;
            if param_attrs.foreign_crate.is_some() && param_attrs.foreign_c_name.is_none() {
                return Err(syn::Error::new_spanned(
                    &pt.pat,
                    "#[ffier(foreign = ...)] requires c_name = \"...\" \
                     (the C typedef name for the foreign handle type)",
                ));
            }
            let foreign_crate_tokens = param_attrs.foreign_crate.as_ref().map(|p| quote! { #p });

            if let Some((trait_name, trait_lifetime_args)) = extract_impl_trait_info(inner_ty) {
                let dispatch = param_attrs.dispatch.unwrap_or_else(|| "auto".to_string());
                return Ok(ParamInfo {
                    name: pi.ident.clone(),
                    kind: ParamKind::ImplTrait {
                        trait_name,
                        dispatch,
                        ref_kind: impl_trait_ref_kind.to_string(),
                        trait_lifetime_args,
                    },
                    types: Some(TypePair {
                        bridge: quote! { *mut core::ffi::c_void },
                        rust: quote! { *mut core::ffi::c_void },
                        foreign_crate: None,
                        foreign_c_name: None,
                    }),
                });
            }

            // Replace Self with concrete type — lifetime-erased for bridge,
            // lifetime-preserving for rust (client codegen needs real lifetimes).
            let param_ty_bridge = match self_ty {
                Some(sty) => {
                    let static_ty = erase_lifetimes(sty);
                    replace_self_type(&pt.ty, &static_ty)
                }
                None => (*pt.ty).clone(),
            };
            let param_ty_rust = match self_ty {
                Some(sty) => replace_self_type(&pt.ty, sty),
                None => (*pt.ty).clone(),
            };

            if is_str_slice(&param_ty_bridge) {
                return Ok(ParamInfo {
                    name: pi.ident.clone(),
                    kind: ParamKind::StrSlice,
                    types: None,
                });
            }

            if let Some(elem_ty) = primitive_slice_elem(&param_ty_bridge) {
                let elem_bridge = ctx.bridge_tokens(elem_ty);
                let elem_rust = match self_ty {
                    Some(sty) => {
                        let replaced = replace_self_type(elem_ty, sty);
                        quote! { #replaced }
                    }
                    None => {
                        quote! { #elem_ty }
                    }
                };
                return Ok(ParamInfo {
                    name: pi.ident.clone(),
                    kind: ParamKind::PrimitiveSlice,
                    types: Some(TypePair {
                        bridge: elem_bridge,
                        rust: elem_rust,
                        foreign_crate: foreign_crate_tokens.clone(),
                        foreign_c_name: param_attrs.foreign_c_name.clone(),
                    }),
                });
            }

            if is_handle_slice(&param_ty_bridge) {
                // Extract the inner type T from &[&T] for bridge resolution
                let Type::Reference(ref_ty) = &param_ty_bridge else {
                    unreachable!()
                };
                let Type::Slice(sl) = &*ref_ty.elem else {
                    unreachable!()
                };
                let Type::Reference(inner_ref) = &*sl.elem else {
                    unreachable!()
                };
                let inner_ty = &*inner_ref.elem;
                let elem_bridge = ctx.bridge_tokens(inner_ty);
                let elem_rust = match self_ty {
                    Some(sty) => {
                        let replaced = replace_self_type(inner_ty, sty);
                        quote! { #replaced }
                    }
                    None => {
                        quote! { #inner_ty }
                    }
                };
                return Ok(ParamInfo {
                    name: pi.ident.clone(),
                    kind: ParamKind::HandleSlice,
                    types: Some(TypePair {
                        bridge: elem_bridge,
                        rust: elem_rust,
                        foreign_crate: foreign_crate_tokens.clone(),
                        foreign_c_name: param_attrs.foreign_c_name.clone(),
                    }),
                });
            }

            let bridge = ctx.bridge_tokens(&param_ty_bridge);
            Ok(ParamInfo {
                name: pi.ident.clone(),
                kind: ParamKind::Regular,
                types: Some(TypePair {
                    bridge,
                    rust: quote! { #param_ty_rust },
                    foreign_crate: foreign_crate_tokens,
                    foreign_c_name: param_attrs.foreign_c_name,
                }),
            })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    // --- Parse return type ---
    let self_ty_static = self_ty.map(erase_lifetimes);
    let foreign_return_tokens = foreign_return.map(|p| quote! { #p });
    let foreign_ret_c_name = foreign_return_c_name.map(|s| s.to_string());

    // Builder detection: method returns Self (only for exported impls, requires a receiver)
    let has_receiver = receiver != Receiver::None;
    let is_builder = if let (Some(sty_static), true) = (&self_ty_static, has_receiver) {
        match &sig.output {
            ReturnType::Default => false,
            ReturnType::Type(_, ty) => {
                let ty = &replace_self_type(ty, sty_static);
                is_self_return(ty, sty_static)
                    || extract_result_types(ty)
                        .is_some_and(|(ok, _)| is_self_return(&ok, sty_static))
            }
        }
    } else {
        false
    };

    let ret = match &sig.output {
        ReturnType::Default => ReturnKind::Void,
        ReturnType::Type(_, ty) => {
            // Bridge path: Self replaced with lifetime-erased type
            let ty_bridge = match &self_ty_static {
                Some(sty) => replace_self_type(ty, sty),
                None => (**ty).clone(),
            };
            // Rust path: Self replaced with original type (lifetimes preserved)
            let ty_rust = match self_ty {
                Some(sty) => replace_self_type(ty, sty),
                None => (**ty).clone(),
            };

            if is_builder && extract_result_types(&ty_bridge).is_none() {
                // Builder returning Self → void in C
                ReturnKind::Void
            } else if let Some(inner_ty) = handle_slice_elem(&ty_bridge) {
                // &[&T] or &[T] where T is an exported handle
                let elem_bridge = ctx.bridge_tokens(inner_ty);
                let inner_ty_rust =
                    handle_slice_elem(&ty_rust).expect("rust type should also be a handle slice");
                // Detect &[T] (direct) vs &[&T] (indirect) by checking
                // whether the slice element is a reference.
                let Type::Reference(ref_ty) = &ty_bridge else {
                    unreachable!()
                };
                let Type::Slice(sl) = &*ref_ty.elem else {
                    unreachable!()
                };
                let direct = !matches!(&*sl.elem, Type::Reference(_));
                ReturnKind::HandleSlice {
                    types: TypePair {
                        bridge: elem_bridge,
                        rust: quote! { #inner_ty_rust },
                        foreign_crate: foreign_return_tokens.clone(),
                        foreign_c_name: foreign_ret_c_name.clone(),
                    },
                    direct,
                }
            } else if let Some((ok_bridge, err)) = extract_result_types(&ty_bridge) {
                let err_ident = type_ident_name(&err);
                let ok_rust = extract_result_types(&ty_rust).map(|(ok, _)| ok);
                let ok_pair = if is_unit_type(&ok_bridge)
                    || (is_builder
                        && self_ty_static
                            .as_ref()
                            .is_some_and(|sty| is_self_return(&ok_bridge, sty)))
                {
                    None
                } else if raw_handle {
                    let erased = erase_lifetimes(&ok_bridge);
                    let rust = ok_rust.as_ref().unwrap_or(&ok_bridge);
                    Some(TypePair {
                        bridge: quote! { #erased },
                        rust: quote! { #rust },
                        foreign_crate: foreign_return_tokens.clone(),
                        foreign_c_name: foreign_ret_c_name.clone(),
                    })
                } else {
                    let bridge = ctx.bridge_tokens(&ok_bridge);
                    let rust = ok_rust.as_ref().unwrap_or(&ok_bridge);
                    Some(TypePair {
                        bridge,
                        rust: quote! { #rust },
                        foreign_crate: foreign_return_tokens.clone(),
                        foreign_c_name: foreign_ret_c_name.clone(),
                    })
                };
                ReturnKind::Result {
                    ok: ok_pair,
                    err_ident,
                }
            } else if raw_handle {
                let erased = erase_lifetimes(&ty_bridge);
                ReturnKind::Value(TypePair {
                    bridge: quote! { #erased },
                    rust: quote! { #ty_rust },
                    foreign_crate: foreign_return_tokens.clone(),
                    foreign_c_name: foreign_ret_c_name.clone(),
                })
            } else {
                let bridge = ctx.bridge_tokens(&ty_bridge);
                ReturnKind::Value(TypePair {
                    bridge,
                    rust: quote! { #ty_rust },
                    foreign_crate: foreign_return_tokens,
                    foreign_c_name: foreign_ret_c_name,
                })
            }
        }
    };

    // --- Original rust_ret for client codegen ---
    let rust_ret = match &sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => {
            let replaced = match self_ty {
                Some(sty) => replace_self_type(ty, sty),
                None => (**ty).clone(),
            };
            Some(quote! { #replaced })
        }
    };

    Ok(Some(MethodInfo {
        name: sig.ident.clone(),
        receiver,
        params,
        ret,
        ffi_name: String::new(),
        is_builder,
        method_lifetimes: sig
            .generics
            .lifetimes()
            .map(|lt| lt.lifetime.ident.clone())
            .collect(),
        doc_lines: extract_doc_comments(attrs),
        rust_ret,
        has_default,
        index: 0,
        raw_handle,
        cfg_predicates: extract_cfg_predicates(attrs),
    }))
}

/// Whether the method is from a trait definition or a concrete impl.
#[derive(Clone, Copy)]
enum MethodMetaKind {
    /// Trait definition method — carries index/has_default/raw_handle.
    Definition,
    /// Concrete method (exported struct impl or trait impl) — carries ffi_name/is_builder.
    Impl,
}

/// Emit metadata tokens for a method list.
fn emit_method_meta(methods: &[MethodInfo], ctx: MethodMetaKind) -> Vec<proc_macro2::TokenStream> {
    methods
        .iter()
        .map(|m| emit_one_method_meta(m, ctx))
        .collect()
}

fn emit_one_method_meta(m: &MethodInfo, ctx: MethodMetaKind) -> proc_macro2::TokenStream {
    let mname = &m.name;
    let doc_tokens: Vec<_> = m.doc_lines.iter().map(|d| quote! { #d }).collect();

    let receiver_ident = match m.receiver {
        Receiver::None => format_ident!("none"),
        Receiver::Ref => format_ident!("r#ref"),
        Receiver::Mut => format_ident!("r#mut"),
        Receiver::Value => format_ident!("value"),
    };

    let method_lt_idents: Vec<_> = m
        .method_lifetimes
        .iter()
        .map(|lt| format_ident!("{}", lt))
        .collect();

    let param_tokens: Vec<_> = m.params.iter().map(|p| {
        let id = &p.name;
        let kind_tokens = match &p.kind {
            ParamKind::Regular => quote! { regular },
            ParamKind::StrSlice => quote! { str_slice },
            ParamKind::HandleSlice => quote! { handle_slice },
            ParamKind::PrimitiveSlice => quote! { primitive_slice },
            ParamKind::ImplTrait { trait_name, dispatch, ref_kind, trait_lifetime_args } => {
                let dispatch_ident = format_ident!("{dispatch}");
                let ref_kind_ident = match ref_kind.as_str() {
                    "ref" => format_ident!("r#ref"),
                    "mut" => format_ident!("r#mut"),
                    other => format_ident!("{other}"),
                };
                let lt_idents: Vec<_> = trait_lifetime_args.iter().map(|lt| format_ident!("{lt}")).collect();
                quote! { impl_trait, trait_name = #trait_name, dispatch = #dispatch_ident, ref_kind = #ref_kind_ident, trait_lifetime_args = [#(#lt_idents),*] }
            }
        };
        let type_tokens = if p.is_impl_trait() {
            quote! {}
        } else {
            match &p.types {
                Some(tp) => {
                    let bt = &tp.bridge;
                    let rt = &tp.rust;
                    let fc = tp.foreign_crate.as_ref().map(|fc| quote! { foreign_crate = (#fc), });
                    let fcn = tp.foreign_c_name.as_ref().map(|n| quote! { foreign_c_name = (#n), });
                    quote! { bridge_type = (#bt), rust_type = (#rt), #fc #fcn }
                }
                None => quote! {},
            }
        };
        quote! { { name = #id, kind = #kind_tokens, #type_tokens } }
    }).collect();

    let ret_tokens = match &m.ret {
        ReturnKind::Void => quote! { void },
        ReturnKind::Value(tp) => {
            let bt = &tp.bridge;
            let rt = &tp.rust;
            let fc = tp
                .foreign_crate
                .as_ref()
                .map(|fc| quote! { foreign_crate = (#fc), });
            let fcn = tp
                .foreign_c_name
                .as_ref()
                .map(|n| quote! { foreign_c_name = (#n), });
            quote! { value(bridge_type = (#bt), rust_type = (#rt), #fc #fcn) }
        }
        ReturnKind::Result { ok, err_ident } => {
            let ok_tokens = match ok {
                None => quote! { ok = void },
                Some(tp) => {
                    let bt = &tp.bridge;
                    let rt = &tp.rust;
                    let fc = tp
                        .foreign_crate
                        .as_ref()
                        .map(|fc| quote! { foreign_crate = (#fc), });
                    let fcn = tp
                        .foreign_c_name
                        .as_ref()
                        .map(|n| quote! { foreign_c_name = (#n), });
                    quote! { ok = some(bridge_type = (#bt), rust_type = (#rt), #fc #fcn) }
                }
            };
            quote! { result(#ok_tokens, err_ident = #err_ident,) }
        }
        ReturnKind::HandleSlice { types: tp, direct } => {
            let bt = &tp.bridge;
            let rt = &tp.rust;
            let fc = tp
                .foreign_crate
                .as_ref()
                .map(|fc| quote! { foreign_crate = (#fc), });
            let fcn = tp
                .foreign_c_name
                .as_ref()
                .map(|n| quote! { foreign_c_name = (#n), });
            if *direct {
                quote! { direct_handle_slice(bridge_type = (#bt), rust_type = (#rt), #fc #fcn) }
            } else {
                quote! { handle_slice(bridge_type = (#bt), rust_type = (#rt), #fc #fcn) }
            }
        }
    };

    let rust_ret_tokens = match &m.rust_ret {
        Some(rt) => quote! { rust_ret = (#rt), },
        None => quote! { rust_ret = (()), },
    };

    let context_tokens = match ctx {
        MethodMetaKind::Definition => {
            let has_default = m.has_default;
            let index = m.index;
            let raw_handle = m.raw_handle;
            quote! {
                method_kind = definition,
                has_default = #has_default,
                index = #index,
                raw_handle = #raw_handle,
            }
        }
        MethodMetaKind::Impl => {
            let ffi_name = &m.ffi_name;
            let is_builder = m.is_builder;
            quote! {
                method_kind = r#impl,
                ffi_name = #ffi_name,
                is_builder = #is_builder,
            }
        }
    };

    let cfg_tokens = match combine_cfg_predicates(&m.cfg_predicates) {
        Some(pred) => quote! { cfg = (#pred), },
        None => quote! {},
    };

    quote! {
        {
            name = #mname,
            doc = [#(#doc_tokens),*],
            receiver = #receiver_ident,
            #context_tokens
            method_lifetimes = [#(#method_lt_idents),*],
            params = [#(#param_tokens),*],
            ret = #ret_tokens,
            #rust_ret_tokens
            #cfg_tokens
        }
    }
}

/// Generate `impl Trait for &mut dyn Trait` — delegates via deref.
/// This lets the bridge pass `&mut dyn Trait` where `impl Trait` is expected,
/// as a dynamic dispatch fallback when enumerating concrete types would be
/// a combinatorial explosion.
fn emit_dyn_dispatch_impl(trait_item: &ItemTrait) -> proc_macro2::TokenStream {
    let trait_name = &trait_item.ident;
    let method_impls: Vec<_> = trait_item
        .items
        .iter()
        .filter_map(|item| {
            let TraitItem::Fn(method) = item else {
                return None;
            };
            let sig = &method.sig;
            if !matches!(sig.inputs.first(), Some(FnArg::Receiver(_))) {
                return None;
            }
            let name = &sig.ident;
            let params: Vec<_> = sig
                .inputs
                .iter()
                .skip(1)
                .filter_map(|arg| {
                    let FnArg::Typed(pt) = arg else { return None };
                    let Pat::Ident(pi) = &*pt.pat else {
                        return None;
                    };
                    Some(pi.ident.clone())
                })
                .collect();
            Some(quote! { #sig { (**self).#name(#(#params),*) } })
        })
        .collect();

    quote! {
        impl #trait_name for &mut dyn #trait_name {
            #(#method_impls)*
        }
    }
}

/// Generate `impl Trait for &mut dyn Trait` for dynamic dispatch fallback.
///
/// `#[ffier::export]` on traits implies `#[ffier::dispatch]` — use this
/// annotation alone when you want dynamic dispatch fallback without
/// exporting the trait's vtable to C.
#[proc_macro_attribute]
pub fn dispatch(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let trait_item = parse_macro_input!(item as ItemTrait);
    let original_trait = trait_item.clone();
    let dyn_impl = emit_dyn_dispatch_impl(&trait_item);

    let output = quote! {
        #original_trait
        #dyn_impl
    };

    output.into()
}

fn implementable_inner(args: ImplementableArgs, trait_item: ItemTrait) -> TokenStream {
    let is_foreign = args.foreign;
    let original_trait = trait_item.clone();
    let export_cfg = &args.export_cfg;

    let trait_name = &trait_item.ident;
    let trait_name_str = trait_name.to_string();
    let trait_snake = camel_to_snake(&trait_name_str);

    // Extract trait-level lifetime params (e.g. `<'a>` from `trait Snapshot<'a>`)
    let trait_lifetime_idents: Vec<&syn::Ident> = trait_item
        .generics
        .lifetimes()
        .map(|lt| &lt.lifetime.ident)
        .collect();

    let vtable_struct_name = format_ident!("{trait_name_str}Vtable");

    let wrapper_name = format_ident!("Vtable{trait_name_str}");
    let wrapper_c_handle_suffix = format!("Vtable{trait_name_str}");

    let helper_mod_name = format_ident!("_ffier_vtable_{trait_snake}");
    let mut ctx = AliasContext::new(helper_mod_name.clone());

    let vtable_methods = match extract_vtable_methods(&trait_item, &args.reserved, &mut ctx) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };
    let own_method_count = vtable_methods.len();
    let bless_tokens = match &args.bless {
        Some(s) => quote::quote! { #s },
        None => quote::quote! { none },
    };
    let no_vtable = args.no_vtable;
    let no_vtable_tokens = if no_vtable {
        quote::quote! { yes }
    } else {
        quote::quote! { no }
    };

    // Build trait path with 'static lifetimes for the wrapper impl
    let trait_generics = &trait_item.generics;
    let trait_ty_generics = {
        let mut g = trait_generics.clone();
        for lt in g.lifetimes_mut() {
            lt.lifetime = syn::Lifetime::new("'static", lt.lifetime.apostrophe);
        }
        let (_, tg, _) = g.split_for_impl();
        tg.to_token_stream()
    };
    // Erase lifetimes only on trait-level generics, not on method parameter types.
    // Method signatures keep their original lifetimes (anonymous `'_` for input
    // positions like `s: &str`) — erasing them to `'static` would make the impl
    // unsatisfiable for callers with shorter borrows.
    let trait_item_erased = {
        let mut t = trait_item.clone();
        struct Eraser;
        impl VisitMut for Eraser {
            fn visit_lifetime_mut(&mut self, lt: &mut syn::Lifetime) {
                *lt = syn::Lifetime::new("'static", lt.apostrophe);
            }
            // Don't descend into method signatures — keep their lifetimes as-is.
            fn visit_trait_item_fn_mut(&mut self, _: &mut syn::TraitItemFn) {}
        }
        Eraser.visit_item_trait_mut(&mut t);
        t
    };

    // Helper: generate the vtable call expression for a method (unwrapping Option).
    // `fallback` is Some(tokens) for defaulted methods, None for required methods.
    //
    // For defaulted methods, we first check whether the handle metadata carries
    // ffier's default-dispatch marker for this method index. If so, we skip
    // vtable dispatch and call the library default directly. This prevents
    // infinite re-entrancy when a client trait default calls through
    // self-dispatch.
    // Helper: generate the vtable call expression for a method.
    // `vtable_struct_ref` is the token stream referencing the vtable struct
    // (e.g. `PushStrVtable` for direct output, or `$crate::PushStrVtable`
    // for use inside a macro_rules! body that expands in another crate).
    // --- Default method extraction ---
    // For methods with default bodies, extract the body into a free helper function
    // and rewrite the trait's default to call it. This allows the VtableXxx impl's
    // None branch to also call the helper, preserving the default behavior when
    // the C side doesn't provide a function pointer.
    //
    // Visitor to replace `self` with `__self` in default method bodies.
    struct SelfReplacer;
    impl SelfReplacer {
        /// Replace `self` idents in a raw TokenStream (for macro bodies
        /// that syn::visit_mut doesn't traverse).
        fn replace_in_token_stream(
            &self,
            ts: proc_macro2::TokenStream,
        ) -> proc_macro2::TokenStream {
            ts.into_iter()
                .map(|tt| match tt {
                    proc_macro2::TokenTree::Ident(ref id) if id == "self" => {
                        proc_macro2::TokenTree::Ident(proc_macro2::Ident::new("__self", id.span()))
                    }
                    proc_macro2::TokenTree::Group(g) => {
                        let inner = self.replace_in_token_stream(g.stream());
                        let mut new_g = proc_macro2::Group::new(g.delimiter(), inner);
                        new_g.set_span(g.span());
                        proc_macro2::TokenTree::Group(new_g)
                    }
                    other => other,
                })
                .collect()
        }
    }
    impl VisitMut for SelfReplacer {
        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            if let syn::Expr::Path(ep) = expr
                && ep.qself.is_none()
                && ep.path.is_ident("self")
            {
                *expr = syn::parse_quote! { __self };
                return;
            }
            syn::visit_mut::visit_expr_mut(self, expr);
        }
        fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
            mac.tokens = self.replace_in_token_stream(mac.tokens.clone());
        }
    }

    // Foreign traits must not have default method bodies — the real defaults
    // live in the foreign crate and can't be replicated here.
    // Exception: raw_handle methods always need their default body because
    // they're not dispatched through the vtable.
    if is_foreign {
        for item in &trait_item.items {
            if let TraitItem::Fn(method) = item {
                let is_raw_handle = vtable_methods
                    .iter()
                    .any(|vm| vm.name == method.sig.ident && vm.raw_handle);
                if method.default.is_some() && !is_raw_handle {
                    return syn::Error::new_spanned(
                        &method.sig.ident,
                        "foreign trait methods must not have default bodies \
                         (the defaults live in the foreign crate)",
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }
    }

    let mut default_helpers: Vec<proc_macro2::TokenStream> = Vec::new();
    // Map from method name → helper fn ident (only for methods with defaults)
    let mut default_helper_names: HashMap<String, syn::Ident> = HashMap::new();

    // Build modified trait: replace default bodies with helper calls
    // and strip #[ffier(index = N)] attributes (consumed by macro).
    let mut modified_trait = original_trait.clone();
    for item in &mut modified_trait.items {
        let TraitItem::Fn(method) = item else {
            continue;
        };
        // Strip all #[ffier(...)] from emitted method attrs (consumed by this macro)
        let is_raw_handle = vtable_methods
            .iter()
            .any(|vm| vm.name == method.sig.ident && vm.raw_handle);
        strip_ffier_attrs(&mut method.attrs);
        let Some(default_block) = &method.default else {
            continue;
        };
        // raw_handle methods keep their default body as-is — they don't
        // go through the vtable helper extraction, and the VtableWrapper
        // uses the trait's default impl directly.
        if is_raw_handle {
            continue;
        }

        let method_name = &method.sig.ident;
        let helper_name = format_ident!("__ffier_default_{trait_name}_{method_name}");
        default_helper_names.insert(method_name.to_string(), helper_name.clone());

        // Extract the default body, rewriting self → __self
        let mut body = default_block.clone();
        SelfReplacer.visit_block_mut(&mut body);

        // Build the helper function signature: same as the trait method but
        // with &self replaced by __self: &(impl Trait + ?Sized)
        let mut helper_sig = method.sig.clone();
        // Remove the receiver and add __self parameter
        let helper_params: Vec<_> = helper_sig
            .inputs
            .iter()
            .filter(|arg| !matches!(arg, syn::FnArg::Receiver(_)))
            .cloned()
            .collect();
        helper_sig.inputs = syn::punctuated::Punctuated::new();
        helper_sig.inputs.push(syn::parse_quote! {
            __self: &(impl #trait_name + ?Sized)
        });
        for p in helper_params {
            helper_sig.inputs.push(p);
        }
        helper_sig.ident = helper_name.clone();

        default_helpers.push(quote! {
            #[doc(hidden)]
            pub #helper_sig #body
        });

        // Rewrite the trait's default body to call the helper
        let params_pass: Vec<_> = method
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(pat_type) = arg
                    && let syn::Pat::Ident(pi) = &*pat_type.pat
                {
                    return Some(pi.ident.clone());
                }
                None
            })
            .collect();
        method.default = Some(syn::parse_quote! {
            { #helper_name(self #(, #params_pass)*) }
        });
    }

    // --- Generate VtableXxx method impls ---
    // These are used inside the @on_library_export macro arm. The vtable struct is
    // a sibling (also in @on_library_export), so bare names resolve correctly.

    // --- Compute max vtable slot for metadata ---
    // The highest slot index that needs to be padded in the vtable struct.
    // This accounts for both method indices and reserved (retired) slots.
    let max_method_index = vtable_methods.iter().map(|m| m.index).max();
    let max_reserved_index = args.reserved.iter().copied().max();
    let max_vtable_slot_val: usize = match (max_method_index, max_reserved_index) {
        (Some(mi), Some(ri)) => mi.max(ri),
        (Some(v), None) | (None, Some(v)) => v,
        (None, None) => 0,
    };

    // --- Metadata emission ---
    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name = format_ident!("__ffier_internal_{trait_snake}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{trait_name}");

    let vtable_method_meta = emit_method_meta(&vtable_methods, MethodMetaKind::Definition);

    // Bake method signatures for the vtable proc macro. These are the
    // original trait method signatures (with lifetimes erased) that the
    // proc macro uses for the trait impl — we can't reconstruct them
    // from metadata alone (e.g. &mut impl PushStr vs impl PushStr).
    let vtable_method_sigs: Vec<proc_macro2::TokenStream> = trait_item_erased
        .items
        .iter()
        .filter_map(|item| {
            let TraitItem::Fn(method) = item else {
                return None;
            };
            let name = &method.sig.ident;
            let vm = vtable_methods.iter().find(|v| v.name == *name)?;
            if vm.raw_handle {
                return None;
            }
            let sig = &method.sig;
            Some(quote! { (#sig) })
        })
        .collect();

    // Serialize default helper mappings for the proc macro
    let default_helper_tokens: Vec<proc_macro2::TokenStream> = default_helper_names
        .iter()
        .map(|(method_name_str, helper_ident)| {
            let method_ident = format_ident!("{method_name_str}");
            quote! { #method_ident => ($crate::#helper_ident) }
        })
        .collect();

    let reserved_lits: Vec<_> = args.reserved.iter().map(|r| quote! { #r }).collect();

    let trait_path_tokens = quote! { $crate::#trait_name };

    let local_type_aliases = ctx.local_type_aliases();

    // Generate &mut dyn Trait dispatch delegation (implies #[ffier::dispatch])
    // For traits with supertraits, we also need to delegate the supertrait methods.
    // Skip for foreign traits (orphan rules: can't impl foreign trait for local type).
    let boxdyn_impl = if is_foreign {
        quote! {}
    } else {
        let has_supertraits = trait_item
            .supertraits
            .iter()
            .any(|b| matches!(b, syn::TypeParamBound::Trait(_)));
        // Traits with `impl Trait` params aren't dyn-compatible
        let has_impl_trait = vtable_methods
            .iter()
            .any(|m| m.params.iter().any(|p| p.is_impl_trait()));
        if !has_supertraits && !has_impl_trait {
            emit_dyn_dispatch_impl(&trait_item)
        } else {
            // TODO: generate supertrait delegation for &mut dyn Trait
            quote! {}
        }
    };

    // For foreign traits (`extern trait`), don't re-emit the trait definition —
    // it already exists in the foreign crate. Only emit the generated plumbing.
    let trait_tokens = if is_foreign {
        quote! {}
    } else {
        quote! { #modified_trait }
    };

    // When no_vtable is set, the @on_library_export arm emits nothing —
    // no vtable struct, no wrapper type, no FfiHandle/FfiType impls.
    // C callers cannot implement this trait; only concrete Rust types dispatch.
    let on_library_export_body = if no_vtable {
        quote! {}
    } else {
        quote! {
            // Vtable struct + trait impl generated by proc macro with
            // handle-type awareness for correct Result conventions.
            //
            // The proc macro output uses bare names (FfiType, FfiHandle, etc.)
            // which are brought into scope by `use $crate::*` inside const _.
            // The vtable struct is emitted OUTSIDE the const block (it needs
            // to be visible for offset_of! from other impls) but its field
            // types use fully-qualified paths that the proc macro embeds
            // from the bridge_type tokens in the metadata.
            ffier::__generate_vtable! {
                vtable_struct = #vtable_struct_name;
                wrapper = #wrapper_name;
                trait_path = (#trait_path_tokens);
                trait_generics = (#trait_ty_generics);
                crate_path = ($crate);
                handles = [$($__handle),*];
                reserved = [#(#reserved_lits),*];
                own_method_count = #own_method_count;
                methods = [#(#vtable_method_meta),*];
                method_sigs = [#(#vtable_method_sigs),*];
                default_helpers = [#(#default_helper_tokens),*];
            }

            #[repr(C)]
            pub struct #wrapper_name {
                pub value: ffier::VtableHandle,
            }

            impl Drop for #wrapper_name {
                fn drop(&mut self) {
                    let drop_field: Option<unsafe extern "C" fn(*mut core::ffi::c_void)> = unsafe {
                        self.value.field_or_none(
                            core::mem::offset_of!(#vtable_struct_name, drop),
                        )
                    };
                    if let Some(drop_fn) = drop_field {
                        unsafe { drop_fn(self.value.user_data as *mut core::ffi::c_void) };
                    }
                }
            }

            impl FfiHandle for #wrapper_name {
                const C_HANDLE_NAME: &str = #wrapper_c_handle_suffix;
                const TYPE_TAG: u32 = $type_tag;
            }

            impl FfiType for #wrapper_name {
                type CRepr = *mut core::ffi::c_void;
                const C_TYPE_NAME: &str = #wrapper_c_handle_suffix;
                const IS_HANDLE: bool = true;
                fn into_c(self) -> *mut core::ffi::c_void {
                    ffier::ffier_handle_new(
                        <Self as FfiHandle>::TYPE_TAG,
                        self,
                    )
                }
                unsafe fn from_c(repr: *mut core::ffi::c_void) -> Self {
                    unsafe { ffier::ffier_handle_consume::<Self>(repr) }
                }
            }
        }
    };

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #(#default_helpers)*

        #cfg_attr
        #trait_tokens

        #cfg_attr
        #boxdyn_impl

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {
            #(#local_type_aliases)*
        }

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            // @on_library_export: generates the vtable struct, wrapper type, its trait impl,
            // Drop, FfiHandle, and FfiType impls. Called by library_definition!
            // with the type tag.
            //
            // The vtable struct and wrapper type are emitted at the crate root of the
            // invoking crate, so orphan rules are satisfied even when the trait is
            // defined in an upstream crate.
            (@on_library_export, $type_tag:expr, [$($__handle:ident),*]) => {
                #on_library_export_body
            };
            // Tagged invocation with path overrides (from library_definition!
            // shim for trait entries): the shim passes wrapper_name,
            // vtable_struct, and trait_path as parenthesized groups so the
            // metadata blob uses the library crate's paths for cross-crate
            // traits. Downstream crates in the chain only need the library
             // crate, not the upstream crate that defined the trait.
            ($prefix:literal, $type_tag:expr,
             ($($wrapper:tt)*), ($($tpath:tt)*),
             $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_trait,
                    trait_name = #trait_name,
                    trait_path = ($($tpath)*),
                    prefix = $prefix,
                    type_tag = $type_tag,
                    wrapper_name = ($($wrapper)*),
                    trait_lifetimes = (#(#trait_lifetime_idents),*),
                    vtable_methods = [#(#vtable_method_meta),*],
                    own_method_count = #own_method_count,
                    max_vtable_slot = #max_vtable_slot_val,
                    bless = #bless_tokens,
                    no_vtable = #no_vtable_tokens,
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

// ===========================================================================
// #[ffier::export] on trait impl blocks — export trait method impls as C functions
// ===========================================================================

fn trait_impl_inner(input: ItemImpl, export_cfg: &ExportCfg) -> TokenStream {
    // Build the output impl block with #[ffier(skip)] attributes stripped.
    let mut clean_impl = input.clone();
    for item in &mut clean_impl.items {
        if let ImplItem::Fn(method) = item {
            strip_ffier_attrs(&mut method.attrs);
        }
    }

    // Extract trait path and name
    let Some((_, trait_path, _)) = &input.trait_ else {
        return syn::Error::new_spanned(&input, "trait_impl requires a trait impl block")
            .to_compile_error()
            .into();
    };
    let trait_last_seg = trait_path
        .segments
        .last()
        .expect("trait path must have segments");
    let trait_name = trait_last_seg.ident.clone();
    let trait_snake = camel_to_snake(&trait_name.to_string());

    // Extract lifetime arguments from the trait path (e.g. 'static from AttachDevice<'static>,
    // or 'a from AttachDevice<'a>). These may differ from the impl's declared generics.
    let trait_lt_args: Vec<String> = match &trait_last_seg.arguments {
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
        _ => Vec::new(),
    };

    // Extract struct type
    let Type::Path(ref struct_type_path) = *input.self_ty else {
        return syn::Error::new_spanned(&input.self_ty, "expected a named struct type")
            .to_compile_error()
            .into();
    };
    let struct_last_seg = struct_type_path
        .path
        .segments
        .last()
        .expect("expected struct name");
    let struct_ident = &struct_last_seg.ident;
    let struct_name = struct_ident.to_string();
    let struct_snake = camel_to_snake(&struct_name);

    // Extract lifetime arguments from the struct type (e.g. 'a from View<'a>).
    // Structs without lifetime params (e.g. Widget) will produce an empty list.
    let struct_lt_args: Vec<String> = match &struct_last_seg.arguments {
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
        _ => Vec::new(),
    };

    let helper_mod_name = format_ident!("_ffier_impl_{trait_snake}_for_{struct_snake}");
    let mut ctx = AliasContext::new(helper_mod_name.clone());

    // Extract methods, skipping any marked with #[ffier(skip)].
    let methods: Vec<MethodInfo> = {
        let mut ms = Vec::new();
        for item in &input.items {
            let ImplItem::Fn(method) = item else { continue };
            let mattrs = match parse_ffier_method_attrs(&method.attrs) {
                Ok(a) => a,
                Err(e) => return e.to_compile_error().into(),
            };
            if mattrs.skip {
                continue;
            }
            // trait_impl methods are concrete overrides, not defaults
            match parse_method_sig(
                &method.sig,
                &method.attrs,
                &mut ctx,
                None,
                false,
                mattrs.raw_handle,
                mattrs.foreign_return.as_ref(),
                mattrs.foreign_return_c_name.as_deref(),
            ) {
                Ok(Some(mut m)) => {
                    m.ffi_name = format!("{}_{}", struct_snake, method.sig.ident);
                    ms.push(m);
                }
                Ok(None) => {}
                Err(e) => return e.to_compile_error().into(),
            }
        }
        ms
    };

    let local_type_aliases = ctx.local_type_aliases();

    let method_meta = emit_method_meta(&methods, MethodMetaKind::Impl);

    let lifetime_idents: Vec<_> = input
        .generics
        .lifetimes()
        .map(|lt| format_ident!("{}", lt.lifetime.ident))
        .collect();

    let counter = MACRO_COUNTER.fetch_add(1, Ordering::SeqCst);
    let internal_macro_name =
        format_ident!("__ffier_internal_{trait_snake}_for_{struct_snake}_{counter}");
    let meta_alias_name = format_ident!("__ffier_meta_{trait_name}_for_{struct_ident}");
    let struct_path_tokens = quote! { $crate::#struct_ident };

    let cfg_pred_tokens = export_cfg.meta_tokens();
    let cfg_attr = export_cfg.cfg_attr();

    let output = quote! {
        #cfg_attr
        #clean_impl

        #cfg_attr
        #[doc(hidden)]
        pub mod #helper_mod_name {
            #(#local_type_aliases)*
        }

        #[doc(hidden)]
        #[macro_export]
        macro_rules! #internal_macro_name {
            // library_definition! passes the trait path explicitly to
            // avoid bare-name collisions at the library crate root.
            ($prefix:literal, ($($tpath:tt)*), $callback:path $(, $($rest:tt)*)?) => {
                $callback! { {
                    @exported_trait_impl,
                    trait_name = #trait_name,
                    struct_name = #struct_ident,
                    struct_path = (#struct_path_tokens),
                    trait_path = ($($tpath)*),
                    prefix = $prefix,
                    lifetimes = (#(#lifetime_idents),*),
                    trait_lifetime_args = [#(#trait_lt_args),*],
                    struct_lifetime_args = [#(#struct_lt_args),*],
                    methods = [#(#method_meta),*],
                    #cfg_pred_tokens
                } $(, $($rest)*)? }
            };
        }

        #[doc(hidden)]
        pub use #internal_macro_name as #meta_alias_name;
    };

    output.into()
}

// ===========================================================================
// ffier::library_definition! — define a library's exported types
// ===========================================================================

/// Define the list of exported types for a library.
///
/// The `library_tag` (1–255) uniquely identifies this library. It is
/// composed into the upper 8 bits of every type's tag, leaving 24 bits
/// for the per-type tag. This ensures that handles from different
/// ffier libraries are never accidentally interchangeable — a type tag
/// mismatch will trigger a panic with a clear diagnostic.
///
/// Every entry (except `TraitName for StructName`) must have an explicit
/// type tag: `Name = N`. Tags must be nonzero and unique across the library.
/// Entries can be bare paths or qualified paths (e.g. `crate::submod::Foo`).
///
/// ```ignore
/// ffier::library_definition!("mylib", library_tag = 1,
///     CalcError = 1,
///     Calculator = 2,
///     crate::submod::TextBuffer = 3,
///     BufferError = 4,
///     trait Processor = 10,
///     trait crate::traits::Fruit = 11,
///     crate::traits::Fruit for crate::types::Apple,
/// );
///
/// // Same crate:
/// ffier::generate_bridge!(local = __ffier_mylib_metadata,
///     schema_output = "../../target/ffier-mylib.json");
/// // Or from a separate cdylib crate:
/// ffier::generate_bridge!(external = mylib::__ffier_mylib_metadata,
///     schema_output = "../../target/ffier-mylib.json");
/// ```
///
/// ## Shared primitives prefix
///
/// When multiple ffier libraries share the same primitive types (Str, Bytes,
/// Result, VtableHandle), set `primitives_prefix` to a common prefix:
///
/// ```ignore
/// // krun_init uses "krun" primitives → KrunStr, KrunResult, etc.
/// ffier::library_definition!("krun_init", library_tag = 2,
///     primitives_prefix = "krun",
///     InitError = 1,
///     Config = 2,
/// );
/// ```
///
/// The C header wraps primitive type definitions in `#ifndef` guards so
/// whichever header is included first defines them. Functions like
/// `str_free` always use the library prefix (`krun_init_str_free`).
///
/// Supports three entry kinds:
/// - `Path = N` — exported struct or error enum with type tag
/// - `trait Path = N` — exported trait with type tag
/// - `TraitPath for StructPath` — trait impl bridge (uses the struct's tag)
///
/// Each annotated type generates a `__ffier_meta_*` alias macro next to the
/// type via `pub use`. This macro resolves those aliases from the given paths
/// and invokes their `@on_library_export` arm to generate type impls at the crate root.
#[proc_macro]
pub fn library_definition(input: TokenStream) -> TokenStream {
    let parsed = parse_macro_input!(input as LibraryInput);
    let prefix_lit = &parsed.prefix;
    let prefix_str = parsed.prefix.value();
    let library_tag = parsed.library_tag;
    let primitives_prefix_lit = parsed.primitives_prefix.as_ref().unwrap_or(&parsed.prefix);

    // For each entry, compute:
    // 1. shim_macros: shim macros that inject the tag into the metadata macro call
    // 2. reexport_invocations: `alias_path!(@on_library_export, ...)` calls
    // 3. shim_names: bare ident of each shim macro (entry macro adds $crate:: or not)
    let mut shim_macros: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut reexport_invocations: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut shim_names: Vec<syn::Ident> = Vec::new();

    // Collect names of external traits (from TaggedTrait entries) so that
    // TraitImpl entries can use the internal alias instead of bare names.
    let external_trait_names: std::collections::HashSet<String> = parsed
        .entries
        .iter()
        .filter_map(|e| {
            if let LibraryEntry::TaggedTrait(path, _) = &e.entry {
                let is_external = path.segments.len() > 1
                    && path.segments.first().is_none_or(|seg| seg.ident != "crate");
                if is_external {
                    return Some(path_last_ident(path).to_string());
                }
            }
            None
        })
        .collect();

    // Collect handle type names — used to determine GLib vs FfierResult
    // convention for Result-returning vtable methods. Includes both
    // plain Tagged types (Widget, Gadget) and VtableFoo wrappers from
    // TaggedTrait entries to match the bridge's handle_types set.
    let handle_type_idents: Vec<syn::Ident> = parsed
        .entries
        .iter()
        .filter_map(|e| match &e.entry {
            LibraryEntry::Tagged(path, _) => Some(path_last_ident(path).clone()),
            LibraryEntry::TaggedTrait(path, _) => {
                let trait_name = path_last_ident(path);
                Some(format_ident!("Vtable{trait_name}"))
            }
            _ => None,
        })
        .collect();

    for entry in &parsed.entries {
        let cfg = entry.cfg.as_ref();
        match &entry.entry {
            LibraryEntry::Tagged(path, tag) => {
                let last_ident = path_last_ident(path);
                let full_tag = (library_tag << 24) | tag;

                // Shim macro that injects the tag into the metadata macro call
                let alias = meta_alias_for_type(path);
                let alias_chain = to_chain_path(&alias);
                let shim_name = format_ident!("__ffier_tagged_{prefix_str}_{last_ident}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        #alias_chain! { $prefix, #full_tag, $callback $(, $($rest)*)? }
                    },
                ));

                shim_names.push(shim_name.clone());

                // @on_library_export generates FfiHandle + FfiType impls — must be
                // gated because these reference the concrete type which may not exist.
                reexport_invocations.push(maybe_cfg_wrap(
                    cfg,
                    quote! { #alias!(@on_library_export, #full_tag, [#(#handle_type_idents),*]); },
                ));

                // Helper module re-export for qualified paths
                let helper_mod_name =
                    format_ident!("_ffier_{}", camel_to_snake(&last_ident.to_string()));
                if path.segments.len() > 1 {
                    let helper_mod_path = replace_last_segment(path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }
            }
            LibraryEntry::TaggedTrait(path, tag) => {
                let last_ident = path_last_ident(path);
                let full_tag = (library_tag << 24) | tag;

                // The upstream metadata macro generates the wrapper type
                // via @on_library_export. The shim passes path overrides
                // (wrapper, vtable struct, trait path) using $crate:: which
                // resolves to the library crate. Downstream crates in the
                // chain only need the library crate, not the upstream crate.
                let alias = meta_alias_for_type(path);
                let wrapper_ident = format_ident!("Vtable{last_ident}");

                // Use `_trait_` in internal names to avoid collisions with
                // user types that share the same last segment (e.g. crate::Error
                // and trait ffier_builtins::Error).
                // Upstream alias — unconditional because the shim references
                // it during chain expansion regardless of whether the feature is on.
                let upstream_alias = format_ident!("__ffier_upstream_trait_{last_ident}");
                reexport_invocations.push(quote! {
                    #[doc(hidden)]
                    pub use #alias as #upstream_alias;
                });

                let is_external = path.segments.len() > 1
                    && path.segments.first().is_none_or(|seg| seg.ident != "crate");
                let trait_reexport = format_ident!("__ffier_reexport_trait_{last_ident}");

                if is_external {
                    // Trait reexport — unconditional (chain needs it).
                    reexport_invocations.push(quote! {
                        #[doc(hidden)]
                        pub use #path as #trait_reexport;
                    });

                    // Helper module re-export — cfg-gated (only exists when feature is on).
                    let trait_snake = camel_to_snake(&last_ident.to_string());
                    let helper_mod_name = format_ident!("_ffier_vtable_{trait_snake}");
                    let helper_mod_path = replace_last_segment(path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }

                // Shim paths: wrapper and trait are at the library root (from @on_library_export)
                let shim_trait = if is_external {
                    quote! { $crate::#trait_reexport }
                } else {
                    quote! { $crate::#last_ident }
                };
                let shim_wrapper = quote! { $crate::#wrapper_ident };

                let shim_name = format_ident!("__ffier_tagged_trait_{prefix_str}_{last_ident}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        $crate::#upstream_alias! { $prefix, #full_tag,
                            (#shim_wrapper),
                            (#shim_trait),
                            $callback $(, $($rest)*)? }
                    },
                ));

                // @on_library_export generates the vtable struct + wrapper type + impls.
                reexport_invocations.push(maybe_cfg_wrap(
                    cfg,
                    quote! { #alias!(@on_library_export, #full_tag, [#(#handle_type_idents),*]); },
                ));
                shim_names.push(shim_name.clone());
            }
            LibraryEntry::Enum(path) => {
                let last_ident = path_last_ident(path);
                let alias = meta_alias_for_type(path);
                let alias_chain = to_chain_path(&alias);

                // Shim macro — no type tag needed for enums
                let shim_name = format_ident!("__ffier_enum_{prefix_str}_{last_ident}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        #alias_chain! { $prefix, 0, $callback $(, $($rest)*)? }
                    },
                ));

                // @on_library_export generates the FfiType impl for the enum
                reexport_invocations.push(maybe_cfg_wrap(
                    cfg,
                    quote! { #alias!(@on_library_export, 0u32, [#(#handle_type_idents),*]); },
                ));
                shim_names.push(shim_name.clone());

                // Helper module re-export for qualified paths
                let helper_mod_name =
                    format_ident!("_ffier_{}", camel_to_snake(&last_ident.to_string()));
                if path.segments.len() > 1 {
                    let helper_mod_path = replace_last_segment(path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }
            }
            LibraryEntry::Bitflags(path) => {
                let last_ident = path_last_ident(path);
                let alias = meta_alias_for_type(path);
                let alias_chain = to_chain_path(&alias);

                // Shim macro — no type tag needed for bitflags
                let shim_name = format_ident!("__ffier_bitflags_{prefix_str}_{last_ident}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        #alias_chain! { $prefix, 0, $callback $(, $($rest)*)? }
                    },
                ));

                // @on_library_export generates the FfiType impl for the bitflags type
                reexport_invocations.push(maybe_cfg_wrap(
                    cfg,
                    quote! { #alias!(@on_library_export, 0u32, [#(#handle_type_idents),*]); },
                ));
                shim_names.push(shim_name.clone());

                // Helper module re-export for qualified paths
                let helper_mod_name =
                    format_ident!("_ffier_{}", camel_to_snake(&last_ident.to_string()));
                if path.segments.len() > 1 {
                    let helper_mod_path = replace_last_segment(path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }
            }
            LibraryEntry::FreeFn(path) => {
                let last_ident = path_last_ident(path);
                let alias = meta_alias_for_type(path);
                let alias_chain = to_chain_path(&alias);

                // Shim macro — no type tag needed for free functions
                let shim_name = format_ident!("__ffier_fn_{prefix_str}_{last_ident}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        #alias_chain! { $prefix, 0, $callback $(, $($rest)*)? }
                    },
                ));

                shim_names.push(shim_name.clone());

                // Helper module re-export for qualified paths
                let helper_mod_name = format_ident!("_ffier_fn_{}", last_ident);
                if path.segments.len() > 1 {
                    let helper_mod_path = replace_last_segment(path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }
            }
            LibraryEntry::TraitImpl {
                trait_path,
                struct_path,
            } => {
                let trait_name = path_last_ident(trait_path);
                let struct_name = path_last_ident(struct_path);
                let alias_ident = format_ident!("__ffier_meta_{trait_name}_for_{struct_name}");
                // The trait_impl alias lives next to the struct (where the impl block is)
                let alias = replace_last_segment(struct_path, &alias_ident);
                let alias_chain = to_chain_path(&alias);

                // Determine the trait path to pass through to the metadata.
                // For external traits, use the internal alias to avoid bare-name
                // collisions (e.g. user's `Error` enum vs builtin `Error` trait).
                // Check both the path itself AND the external_trait_names set
                // (handles `Error for TestError` where the trait is bare `Error`
                // but was registered as `trait ffier_builtins::Error = 25`).
                let is_external = external_trait_names.contains(&trait_name.to_string())
                    || (trait_path.segments.len() > 1
                        && trait_path
                            .segments
                            .first()
                            .is_none_or(|seg| seg.ident != "crate"));
                let resolved_trait_path = if is_external {
                    let reexport = format_ident!("__ffier_reexport_trait_{trait_name}");
                    quote! { $crate::#reexport }
                } else {
                    quote! { $crate::#trait_name }
                };

                // Shim macro that passes the resolved trait path to the
                // trait_impl metadata macro.
                let shim_name =
                    format_ident!("__ffier_trait_impl_{prefix_str}_{trait_name}_for_{struct_name}");
                shim_macros.push(emit_library_shim(
                    &shim_name,
                    quote! {
                        #alias_chain! { $prefix, (#resolved_trait_path),
                            $callback $(, $($rest)*)? }
                    },
                ));
                shim_names.push(shim_name.clone());

                // Helper module re-export for qualified paths
                let trait_snake = camel_to_snake(&trait_name.to_string());
                let struct_snake = camel_to_snake(&struct_name.to_string());
                let helper_mod_name = format_ident!("_ffier_impl_{trait_snake}_for_{struct_snake}");
                if struct_path.segments.len() > 1 {
                    let helper_mod_path = replace_last_segment(struct_path, &helper_mod_name);
                    reexport_invocations.push(maybe_cfg_wrap(
                        cfg,
                        quote! {
                            #[doc(hidden)]
                            pub use #helper_mod_path;
                        },
                    ));
                }
            }
        }
    }

    if shim_names.is_empty() {
        return quote! { compile_error!("library_definition! requires at least one type"); }.into();
    }

    let first = &shim_names[0];
    let rest = &shim_names[1..];

    let entry_macro_name = format_ident!("__ffier_{prefix_str}_metadata");

    let output = quote! {
        #[doc(hidden)]
        pub trait FfiHandle {
            const C_HANDLE_NAME: &'static str;
            const TYPE_TAG: u32;
        }

        #[doc(hidden)]
        pub trait FfiType {
            type CRepr;
            const C_TYPE_NAME: &'static str;
            const IS_HANDLE: bool;
            fn into_c(self) -> Self::CRepr;
            /// Reconstruct a value from its C representation.
            ///
            /// # Safety
            /// For handle types, `repr` must be a valid handle pointer.
            unsafe fn from_c(repr: Self::CRepr) -> Self;
        }

        /// Produce a CRepr from `&self` without consuming.
        /// Used by error payload getters to borrow data from an error handle.
        /// Only implemented for types that appear as error payload fields
        /// (primitives, enums, bitflags, strings, byte slices, paths).
        #[doc(hidden)]
        pub trait FfiBorrow: FfiType {
            fn borrow_as_c(&self) -> Self::CRepr;
        }

        #[doc(hidden)]
        pub use ffier::FfiError;

        macro_rules! __ffier_impl_ffi_identity {
            ($($rust_ty:ty => $c_name:expr),* $(,)?) => {
                $(impl FfiType for $rust_ty {
                    type CRepr = $rust_ty;
                    const C_TYPE_NAME: &'static str = $c_name;
                    const IS_HANDLE: bool = false;
                    fn into_c(self) -> Self { self }
                    unsafe fn from_c(repr: Self) -> Self { repr }
                }
                impl FfiBorrow for $rust_ty {
                    fn borrow_as_c(&self) -> Self { *self }
                })*
            };
        }
        __ffier_impl_ffi_identity! {
            i8 => "int8_t", i16 => "int16_t", i32 => "int32_t", i64 => "int64_t",
            u8 => "uint8_t", u16 => "uint16_t", u32 => "uint32_t", u64 => "uint64_t",
            isize => "ssize_t", usize => "size_t", bool => "bool",
        }

        // Opaque raw pointers — passed through without transformation.
        impl FfiType for *mut core::ffi::c_void {
            type CRepr = *mut core::ffi::c_void;
            const C_TYPE_NAME: &'static str = "void*";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> Self { self }
            unsafe fn from_c(repr: Self) -> Self { repr }
        }

        impl FfiType for *const core::ffi::c_void {
            type CRepr = *const core::ffi::c_void;
            const C_TYPE_NAME: &'static str = "const void*";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> Self { self }
            unsafe fn from_c(repr: Self) -> Self { repr }
        }

        impl FfiType for &str {
            type CRepr = ffier::FfierBytes;
            const C_TYPE_NAME: &'static str = "FfierStr";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> ffier::FfierBytes { unsafe { ffier::FfierBytes::from_str(self) } }
            unsafe fn from_c(repr: ffier::FfierBytes) -> Self {
                unsafe {
                    let bytes = core::slice::from_raw_parts(repr.data, repr.len);
                    core::str::from_utf8_unchecked(bytes)
                }
            }
        }
        impl FfiBorrow for &str {
            fn borrow_as_c(&self) -> ffier::FfierBytes { unsafe { ffier::FfierBytes::from_str(self) } }
        }

        impl<'a> FfiType for Option<&'a str> {
            type CRepr = ffier::FfierBytes;
            const C_TYPE_NAME: &'static str = "FfierStr";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> ffier::FfierBytes {
                match self {
                    Some(s) => unsafe { ffier::FfierBytes::from_str(s) },
                    None => ffier::FfierBytes::EMPTY,
                }
            }
            unsafe fn from_c(repr: ffier::FfierBytes) -> Self {
                if repr.data.is_null() {
                    None
                } else {
                    unsafe {
                        let bytes = core::slice::from_raw_parts(repr.data, repr.len);
                        Some(core::str::from_utf8_unchecked(bytes))
                    }
                }
            }
        }

        impl FfiType for Box<str> {
            type CRepr = ffier::FfierBytes;
            const C_TYPE_NAME: &'static str = "FfierStr";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> ffier::FfierBytes {
                let leaked: &mut str = Box::leak(self);
                ffier::FfierBytes { data: leaked.as_mut_ptr() as *const u8, len: leaked.len() }
            }
            unsafe fn from_c(repr: ffier::FfierBytes) -> Self {
                unsafe {
                    let slice = core::slice::from_raw_parts_mut(repr.data as *mut u8, repr.len);
                    Box::from_raw(core::str::from_utf8_unchecked_mut(slice))
                }
            }
        }
        impl FfiBorrow for Box<str> {
            fn borrow_as_c(&self) -> ffier::FfierBytes {
                ffier::FfierBytes { data: self.as_ptr(), len: self.len() }
            }
        }

        impl FfiType for &[u8] {
            type CRepr = ffier::FfierBytes;
            const C_TYPE_NAME: &'static str = "FfierBytes";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> ffier::FfierBytes { unsafe { ffier::FfierBytes::from_bytes(self) } }
            unsafe fn from_c(repr: ffier::FfierBytes) -> Self {
                unsafe {
                    if repr.data.is_null() { &[] }
                    else { core::slice::from_raw_parts(repr.data, repr.len) }
                }
            }
        }

        #[cfg(unix)]
        impl FfiType for &std::path::Path {
            type CRepr = ffier::FfierBytes;
            const C_TYPE_NAME: &'static str = "FfierPath";
            const IS_HANDLE: bool = false;
            fn into_c(self) -> ffier::FfierBytes { unsafe { ffier::FfierBytes::from_path(self) } }
            unsafe fn from_c(repr: ffier::FfierBytes) -> Self {
                use std::os::unix::ffi::OsStrExt;
                unsafe {
                    let bytes = core::slice::from_raw_parts(repr.data, repr.len);
                    std::path::Path::new(std::ffi::OsStr::from_bytes(bytes))
                }
            }
        }

        #[cfg(unix)]
        const _: () = {
            use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
            impl FfiType for OwnedFd {
                type CRepr = i32;
                const C_TYPE_NAME: &'static str = "int";
                const IS_HANDLE: bool = false;
                fn into_c(self) -> i32 { self.into_raw_fd() }
                unsafe fn from_c(fd: i32) -> Self { unsafe { OwnedFd::from_raw_fd(fd) } }
            }
            impl<'a> FfiType for BorrowedFd<'a> {
                type CRepr = i32;
                const C_TYPE_NAME: &'static str = "int";
                const IS_HANDLE: bool = false;
                fn into_c(self) -> i32 { self.as_raw_fd() }
                unsafe fn from_c(fd: i32) -> Self { unsafe { BorrowedFd::borrow_raw(fd) } }
            }
            impl<'a> FfiType for Option<BorrowedFd<'a>> {
                type CRepr = i32;
                const C_TYPE_NAME: &'static str = "int";
                const IS_HANDLE: bool = false;
                fn into_c(self) -> i32 {
                    match self {
                        Some(fd) => fd.as_raw_fd(),
                        None => -1,
                    }
                }
                unsafe fn from_c(fd: i32) -> Self {
                    if fd < 0 { None } else { Some(unsafe { BorrowedFd::borrow_raw(fd) }) }
                }
            }
            impl FfiType for Option<OwnedFd> {
                type CRepr = i32;
                const C_TYPE_NAME: &'static str = "int";
                const IS_HANDLE: bool = false;
                fn into_c(self) -> i32 {
                    match self {
                        Some(fd) => fd.into_raw_fd(),
                        None => -1,
                    }
                }
                unsafe fn from_c(fd: i32) -> Self {
                    if fd < 0 { None } else { Some(unsafe { OwnedFd::from_raw_fd(fd) }) }
                }
            }
        };

        impl<T: FfiHandle + 'static> FfiType for &T {
            type CRepr = *mut core::ffi::c_void;
            const C_TYPE_NAME: &'static str = T::C_HANDLE_NAME;
            const IS_HANDLE: bool = true;
            fn into_c(self) -> *mut core::ffi::c_void { unimplemented!("&T into_c") }
            unsafe fn from_c(repr: *mut core::ffi::c_void) -> Self { unsafe { ffier::ffier_handle_borrow::<T>(repr) } }
        }

        impl<T: FfiHandle + 'static> FfiType for &mut T {
            type CRepr = *mut core::ffi::c_void;
            const C_TYPE_NAME: &'static str = T::C_HANDLE_NAME;
            const IS_HANDLE: bool = true;
            fn into_c(self) -> *mut core::ffi::c_void { unimplemented!("&mut T into_c") }
            unsafe fn from_c(repr: *mut core::ffi::c_void) -> Self { unsafe { ffier::ffier_handle_borrow_mut::<T>(repr) } }
        }

        // Create helper modules at the crate root and invoke @on_library_export arms.
        #(#reexport_invocations)*

        #(#shim_macros)*

        // Chain accumulator — shared by both `@cross_crate` and `@local`
        // metadata macro arms. The `$chain` parameter is how the chain
        // references itself: cross-crate passes `$crate::__ffier_chain`,
        // local passes the bare name `__ffier_chain` (rust#52234).
        #[doc(hidden)]
        #[macro_export]
        macro_rules! __ffier_chain {
            // Recursive: append metadata, call next shim
            ({ $($meta:tt)* }, $prefix:literal, $chain:path, $final_cb:path,
             $schema_output:literal,
             [$($acc:tt)*], [$next:path $(, $($remaining:path),*)?]) => {
                $next! { $prefix, $chain,
                    $prefix, $chain, $final_cb,
                    $schema_output,
                    [$($acc)* { $($meta)* }],
                    [$($($remaining),*)?]
                }
            };
            // Base case: all metadata accumulated, invoke bridge generator
            ({ $($meta:tt)* }, $prefix:literal, $chain:path, $final_cb:path,
             $schema_output:literal,
             [$($acc:tt)*], []) => {
                $final_cb! { @lib_crate = $crate; @primitives_prefix = #primitives_prefix_lit; @schema_output = $schema_output; $($acc)* { $($meta)* } }
            };
        }

        /// Metadata entry point for bridge generation.
        ///
        /// Not intended for direct use — call via [`ffier::generate_bridge!`]:
        /// ```ignore
        /// ffier::generate_bridge!(local = __ffier_{prefix}_metadata, schema_output = "target/ffier-{prefix}.json");
        /// ffier::generate_bridge!(external = mylib::__ffier_{prefix}_metadata, schema_output = "target/ffier-{prefix}.json");
        /// ```
        #[macro_export]
        macro_rules! #entry_macro_name {
            (@cross_crate, $schema_output:literal) => {
                $crate::#first! { #prefix_lit, $crate::__ffier_chain,
                    #prefix_lit, $crate::__ffier_chain, ffier::__generate_bridge,
                    $schema_output,
                    [],
                    [#($crate::#rest),*]
                }
            };
            (@local, $schema_output:literal) => {
                #first! { #prefix_lit, __ffier_chain,
                    #prefix_lit, __ffier_chain, ffier::__generate_bridge,
                    $schema_output,
                    [],
                    [#(#rest),*]
                }
            };
        }
    };

    output.into()
}

// ---------------------------------------------------------------------------
// library_definition! helpers
// ---------------------------------------------------------------------------

/// Build the `__ffier_meta_*` alias path for a plain type or trait path.
/// Replaces the last segment: `a::b::Foo` → `a::b::__ffier_meta_Foo`.
fn meta_alias_for_type(path: &syn::Path) -> syn::Path {
    let name = path_last_ident(path);
    let alias_ident = format_ident!("__ffier_meta_{name}");
    replace_last_segment(path, &alias_ident)
}

/// Convert an alias path to a chain-usable path.
///
/// - Bare ident `__ffier_meta_Foo` → `$crate::__ffier_meta_Foo`
/// - `crate::a::b::__ffier_meta_Foo` → `$crate::a::b::__ffier_meta_Foo`
/// - `other_crate::__ffier_meta_Foo` → `other_crate::__ffier_meta_Foo` (as-is)
fn to_chain_path(path: &syn::Path) -> proc_macro2::TokenStream {
    let first_seg = &path.segments.first().unwrap().ident;

    if path.segments.len() == 1 {
        // Bare ident — same crate
        quote! { $crate::#path }
    } else if first_seg == "crate" {
        // crate::a::b::X → $crate::a::b::X
        let without_crate: syn::punctuated::Punctuated<syn::PathSegment, Token![::]> =
            path.segments.iter().skip(1).cloned().collect();
        quote! { $crate::#without_crate }
    } else {
        // External crate path — use as-is
        quote! { #path }
    }
}

/// Extract the last identifier from a path.
fn path_last_ident(path: &syn::Path) -> &syn::Ident {
    &path.segments.last().expect("path must have segments").ident
}

/// Replace the last segment of a path with a new ident.
fn replace_last_segment(path: &syn::Path, new_last: &syn::Ident) -> syn::Path {
    let mut result = path.clone();
    let last = result.segments.last_mut().unwrap();
    last.ident = new_last.clone();
    last.arguments = syn::PathArguments::None;
    result
}

// ---------------------------------------------------------------------------
// library_definition! parsing
// ---------------------------------------------------------------------------

struct LibraryInput {
    prefix: LitStr,
    library_tag: u32,
    /// Optional override for primitive type names (Str, Bytes, Result, etc.).
    /// When set, primitive C types use this prefix instead of the library prefix.
    /// For example, `primitives_prefix = "krun"` produces `KrunStr` even if the
    /// library prefix is `"krun_init"`.
    primitives_prefix: Option<LitStr>,
    entries: Vec<ConditionalLibraryEntry>,
}

struct ConditionalLibraryEntry {
    cfg: Option<LibraryEntryCfg>,
    entry: LibraryEntry,
}

struct LibraryEntryCfg {
    predicates: Vec<syn::Meta>,
}

impl LibraryEntryCfg {
    fn enabled_attr(&self) -> proc_macro2::TokenStream {
        let predicate = self.predicate_tokens();
        quote! { #[cfg(#predicate)] }
    }

    fn predicate_tokens(&self) -> proc_macro2::TokenStream {
        if self.predicates.len() == 1 {
            let predicate = &self.predicates[0];
            quote! { #predicate }
        } else {
            let predicates = &self.predicates;
            quote! { all(#(#predicates),*) }
        }
    }
}

enum LibraryEntry {
    /// A plain type or error with an explicit type tag: `Path = N`
    Tagged(syn::Path, u32),
    /// An implementable trait with an explicit type tag: `trait Path = N`
    TaggedTrait(syn::Path, u32),
    /// A trait impl bridge: `TraitPath for StructPath`
    TraitImpl {
        trait_path: syn::Path,
        struct_path: syn::Path,
    },
    /// An enum constant type (no type tag, value type): `enum Path`
    Enum(syn::Path),
    /// A bitflags type (no type tag, value type): `bitflags Path`
    Bitflags(syn::Path),
    /// A free function: `fn Path`
    FreeFn(syn::Path),
}

/// Parse a `= N` type tag after an identifier. Returns the tag value.
/// Rejects tag 0 (reserved for success / no-type) and tags that don't fit
/// in 24 bits (the upper 8 bits are reserved for the library tag).
fn parse_type_tag(input: syn::parse::ParseStream) -> syn::Result<u32> {
    input.parse::<Token![=]>()?;
    let tag_lit: syn::LitInt = input.parse()?;
    let tag = tag_lit.base10_parse::<u32>()?;
    if tag == 0 {
        return Err(syn::Error::new(
            tag_lit.span(),
            "type tag must be nonzero (0 is reserved)",
        ));
    }
    if tag > 0x00FF_FFFF {
        return Err(syn::Error::new(
            tag_lit.span(),
            "type tag must fit in 24 bits (max 16777215); \
             the upper 8 bits are reserved for library_tag",
        ));
    }
    Ok(tag)
}

fn parse_library_entry_cfg(input: syn::parse::ParseStream) -> syn::Result<Option<LibraryEntryCfg>> {
    let mut predicates = Vec::new();

    for attr in input.call(syn::Attribute::parse_outer)? {
        if !attr.path().is_ident("cfg") {
            return Err(syn::Error::new_spanned(
                attr,
                "library_definition! entries only support outer #[cfg(...)] attributes",
            ));
        }
        predicates.push(attr.parse_args::<syn::Meta>()?);
    }

    if predicates.is_empty() {
        Ok(None)
    } else {
        Ok(Some(LibraryEntryCfg { predicates }))
    }
}

fn maybe_cfg_wrap(
    cfg: Option<&LibraryEntryCfg>,
    tokens: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    if let Some(cfg) = cfg {
        let attr = cfg.enabled_attr();
        quote! {
            #attr
            #tokens
        }
    } else {
        tokens
    }
}

fn emit_library_shim(
    shim_name: &syn::Ident,
    on_body: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    // The shim always forwards unconditionally. The metadata macro
    // always exists (emitted by the proc macro regardless of #[cfg]).
    // The cfg predicate is carried inside the metadata blob, not on the shim.
    // The @on_library_export invocations (FfiType impls, vtable generation)
    // are separately gated via maybe_cfg_wrap in library_definition!.
    quote! {
        #[doc(hidden)]
        #[macro_export]
        macro_rules! #shim_name {
            ($prefix:literal, $callback:path $(, $($rest:tt)*)?) => {
                #on_body
            };
        }
    }
}

impl Parse for LibraryInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let prefix: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;

        // Parse required `library_tag = N`
        let lt_ident: syn::Ident = input.parse()?;
        if lt_ident != "library_tag" {
            return Err(syn::Error::new(
                lt_ident.span(),
                "expected `library_tag = N` (required, 1..=255)",
            ));
        }
        input.parse::<Token![=]>()?;
        let lt_lit: syn::LitInt = input.parse()?;
        let library_tag = lt_lit.base10_parse::<u32>()?;
        if library_tag == 0 {
            return Err(syn::Error::new(
                lt_lit.span(),
                "library_tag must be nonzero (0 would make tags \
                 indistinguishable from a library without a tag)",
            ));
        }
        if library_tag > 255 {
            return Err(syn::Error::new(
                lt_lit.span(),
                "library_tag must fit in 8 bits (max 255)",
            ));
        }
        input.parse::<Token![,]>()?;

        // Parse optional `primitives_prefix = "..."`.
        // Must appear immediately after library_tag if present.
        let primitives_prefix = if input.peek(syn::Ident) {
            let fork = input.fork();
            if fork
                .parse::<syn::Ident>()
                .is_ok_and(|id| id == "primitives_prefix")
            {
                let _: syn::Ident = input.parse()?; // consume "primitives_prefix"
                input.parse::<Token![=]>()?;
                let pp: LitStr = input.parse()?;
                input.parse::<Token![,]>()?;
                Some(pp)
            } else {
                None
            }
        } else {
            None
        };

        let mut entries = Vec::new();
        while !input.is_empty() {
            let cfg = parse_library_entry_cfg(input)?;
            let entry = if input.peek(Token![trait]) {
                // `trait Path = N`
                input.parse::<Token![trait]>()?;
                let path: syn::Path = input.parse()?;
                let tag = parse_type_tag(input)?;
                LibraryEntry::TaggedTrait(path, tag)
            } else if input.peek(Token![enum]) {
                // `enum Path`
                input.parse::<Token![enum]>()?;
                let path: syn::Path = input.parse()?;
                LibraryEntry::Enum(path)
            } else if input.peek(syn::Ident)
                && input
                    .fork()
                    .parse::<syn::Ident>()
                    .is_ok_and(|id| id == "bitflags")
            {
                // `bitflags Path`
                let _: syn::Ident = input.parse()?;
                let path: syn::Path = input.parse()?;
                LibraryEntry::Bitflags(path)
            } else if input.peek(Token![fn]) {
                // `fn Path`
                input.parse::<Token![fn]>()?;
                let path: syn::Path = input.parse()?;
                LibraryEntry::FreeFn(path)
            } else {
                let first: syn::Path = input.parse()?;
                if input.peek(Token![for]) {
                    // `TraitPath for StructPath`
                    input.parse::<Token![for]>()?;
                    let second: syn::Path = input.parse()?;
                    LibraryEntry::TraitImpl {
                        trait_path: first,
                        struct_path: second,
                    }
                } else {
                    // `Path = N`
                    let tag = parse_type_tag(input)?;
                    LibraryEntry::Tagged(first, tag)
                }
            };
            entries.push(ConditionalLibraryEntry { cfg, entry });
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(LibraryInput {
            prefix,
            library_tag,
            primitives_prefix,
            entries,
        })
    }
}

// ===========================================================================
// #[ffier::reexport] — re-export ffier metadata alongside types (planned)
// ===========================================================================

// ===========================================================================
// __generate_vtable — proc macro for vtable struct + VtableFoo trait impl
// ===========================================================================
//
// Called from the @on_library_export arm of #[implementable] with handle type names
// from library_definition!. Generates:
// 1. #[repr(C)] pub struct VtableStruct { drop, method fields... }
// 2. const _: () = { use $crate::*; impl Trait for Wrapper { methods... } };
//
// Result-returning methods use GLib-style (HandleOrNull) when the ok type's
// name matches one of the handle names, otherwise OutParam (FfierResult).

/// Input format:
/// ```ignore
/// vtable_struct = Ident;
/// wrapper = Ident;
/// trait_path = (token tokens);
/// trait_generics = (token tokens);  // e.g. <'static> or empty
/// crate_path = (token tokens);      // typically $crate
/// handles = [Ident, Ident, ...];
/// reserved = [N, N, ...];
/// methods = [ {method_meta}, ... ];
/// default_helpers = [ method_name => (path tokens), ... ];
/// ```
struct GenerateVtableInput {
    vtable_struct: syn::Ident,
    wrapper: syn::Ident,
    trait_path: proc_macro2::TokenStream,
    trait_generics: proc_macro2::TokenStream,
    crate_path: proc_macro2::TokenStream,
    handles: Vec<syn::Ident>,
    reserved: Vec<usize>,
    own_method_count: usize,
    methods: Vec<proc_macro2::TokenStream>,
    method_sigs: Vec<syn::Signature>,
    default_helpers: Vec<(syn::Ident, proc_macro2::TokenStream)>,
}

impl Parse for GenerateVtableInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut vtable_struct = None;
        let mut wrapper = None;
        let mut trait_path = None;
        let mut trait_generics = None;
        let mut crate_path = None;
        let mut handles = Vec::new();
        let mut own_method_count = 0usize;
        let mut reserved = Vec::new();
        let mut methods: Vec<proc_macro2::TokenStream> = Vec::new();
        let mut method_sigs: Vec<syn::Signature> = Vec::new();
        let mut default_helpers = Vec::new();

        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "vtable_struct" => {
                    vtable_struct = Some(input.parse::<syn::Ident>()?);
                    input.parse::<Token![;]>()?;
                }
                "wrapper" => {
                    wrapper = Some(input.parse::<syn::Ident>()?);
                    input.parse::<Token![;]>()?;
                }
                "trait_path" => {
                    let content;
                    syn::parenthesized!(content in input);
                    trait_path = Some(content.parse::<proc_macro2::TokenStream>()?);
                    input.parse::<Token![;]>()?;
                }
                "trait_generics" => {
                    let content;
                    syn::parenthesized!(content in input);
                    trait_generics = Some(content.parse::<proc_macro2::TokenStream>()?);
                    input.parse::<Token![;]>()?;
                }
                "crate_path" => {
                    let content;
                    syn::parenthesized!(content in input);
                    crate_path = Some(content.parse::<proc_macro2::TokenStream>()?);
                    input.parse::<Token![;]>()?;
                }
                "handles" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        handles.push(content.parse::<syn::Ident>()?);
                        if !content.is_empty() {
                            content.parse::<Token![,]>()?;
                        }
                    }
                    input.parse::<Token![;]>()?;
                }

                "own_method_count" => {
                    own_method_count = input.parse::<syn::LitInt>()?.base10_parse()?;
                    input.parse::<Token![;]>()?;
                }
                "method_sigs" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        let inner;
                        syn::parenthesized!(inner in content);
                        method_sigs.push(inner.parse::<syn::Signature>()?);
                        if !content.is_empty() {
                            let _ = content.parse::<Token![,]>();
                        }
                    }
                    input.parse::<Token![;]>()?;
                }
                "reserved" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        reserved.push(content.parse::<syn::LitInt>()?.base10_parse()?);
                        if !content.is_empty() {
                            content.parse::<Token![,]>()?;
                        }
                    }
                    input.parse::<Token![;]>()?;
                }
                "methods" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        // Keep the braces — MetaMethod::parse expects them
                        methods.push(
                            content
                                .parse::<proc_macro2::TokenTree>()?
                                .into_token_stream(),
                        );
                        if !content.is_empty() {
                            let _ = content.parse::<Token![,]>();
                        }
                    }
                    input.parse::<Token![;]>()?;
                }
                "default_helpers" => {
                    let content;
                    syn::bracketed!(content in input);
                    while !content.is_empty() {
                        let name: syn::Ident = content.parse()?;
                        content.parse::<Token![=>]>()?;
                        let path_content;
                        syn::parenthesized!(path_content in content);
                        let path = path_content.parse::<proc_macro2::TokenStream>()?;
                        default_helpers.push((name, path));
                        if !content.is_empty() {
                            let _ = content.parse::<Token![,]>();
                        }
                    }
                    input.parse::<Token![;]>()?;
                }

                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown key `{other}` in __generate_vtable"),
                    ));
                }
            }
        }

        Ok(GenerateVtableInput {
            vtable_struct: vtable_struct.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing vtable_struct")
            })?,
            wrapper: wrapper.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing wrapper")
            })?,
            trait_path: trait_path.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing trait_path")
            })?,
            trait_generics: trait_generics.unwrap_or_default(),
            crate_path: crate_path.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing crate_path")
            })?,
            handles,
            reserved,
            own_method_count,
            methods,
            method_sigs,
            default_helpers,
        })
    }
}

/// Build the C function pointer signature for a vtable method:
/// `(param_types, return_type)`. Used for both vtable struct fields
/// and the trait impl's function pointer casts.
/// Returns `(param_types, return_type, ok_is_handle)`.
fn vtable_c_fn_sig(
    m: &crate::meta::MetaMethod,
    handle_names: &std::collections::HashSet<String>,
) -> (
    Vec<proc_macro2::TokenStream>,
    proc_macro2::TokenStream,
    bool,
) {
    let mut param_types = vec![quote! { *mut core::ffi::c_void }];
    for p in &m.params {
        match &p.kind {
            crate::meta::MetaParamKind::ImplTrait { .. } => {
                param_types.push(quote! { *mut core::ffi::c_void });
            }
            crate::meta::MetaParamKind::StrSlice => {
                param_types.push(quote! { *const ffier::FfierBytes });
                param_types.push(quote! { usize });
            }
            crate::meta::MetaParamKind::HandleSlice(_) => {
                param_types.push(quote! { *const *mut core::ffi::c_void });
                param_types.push(quote! { usize });
            }
            crate::meta::MetaParamKind::PrimitiveSlice(tp) => {
                let bt = &tp.bridge_type;
                param_types.push(quote! { *const <#bt as FfiType>::CRepr });
                param_types.push(quote! { usize });
            }
            crate::meta::MetaParamKind::Regular(tp) => {
                let bt = &tp.bridge_type;
                param_types.push(quote! { <#bt as FfiType>::CRepr });
            }
        }
    }

    let ok_is_handle = matches!(&m.ret, crate::meta::MetaReturn::Result { ok: Some(_), .. })
        && crate::meta::is_result_ok_handle(&m.rust_ret, handle_names);

    let fn_ret = match &m.ret {
        crate::meta::MetaReturn::Void => quote! {},
        crate::meta::MetaReturn::Value(tp) => {
            let bt = &tp.bridge_type;
            quote! { -> <#bt as FfiType>::CRepr }
        }
        crate::meta::MetaReturn::HandleSlice { .. } => {
            quote! { -> ffier::FfierObjectArray }
        }
        crate::meta::MetaReturn::Result { ok, .. } => {
            if ok_is_handle {
                param_types.push(quote! { *mut *mut core::ffi::c_void });
                quote! { -> *mut core::ffi::c_void }
            } else {
                if let Some(ok_tp) = ok {
                    let bt = &ok_tp.bridge_type;
                    param_types.push(quote! { *mut <#bt as FfiType>::CRepr });
                }
                param_types.push(quote! { *mut *mut core::ffi::c_void });
                quote! { -> ffier::FfierResult }
            }
        }
    };

    (param_types, fn_ret, ok_is_handle)
}

#[doc(hidden)]
#[proc_macro]
pub fn __generate_vtable(input: TokenStream) -> TokenStream {
    let inp = parse_macro_input!(input as GenerateVtableInput);
    let vtable_struct = &inp.vtable_struct;
    let wrapper = &inp.wrapper;
    let trait_path = &inp.trait_path;
    let trait_generics = &inp.trait_generics;
    let crate_path = &inp.crate_path;
    let handle_names: std::collections::HashSet<String> =
        inp.handles.iter().map(|h| h.to_string()).collect();

    let default_helper_map: HashMap<String, &proc_macro2::TokenStream> = inp
        .default_helpers
        .iter()
        .map(|(name, path)| (name.to_string(), path))
        .collect();

    // Parse each method's metadata using the existing MetaMethod parser
    let parsed_methods: Vec<crate::meta::MetaMethod> = inp
        .methods
        .iter()
        .map(|tokens| {
            syn::parse2::<crate::meta::MetaMethod>(tokens.clone())
                .expect("failed to parse method metadata in __generate_vtable")
        })
        .collect();

    // --- Vtable struct fields ---
    let mut vtable_fields: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut next_slot = 0usize;

    // Sort methods by index for vtable layout
    let mut sorted_methods: Vec<&crate::meta::MetaMethod> = parsed_methods.iter().collect();
    sorted_methods.sort_by_key(|m| m.index());

    for m in &sorted_methods {
        if m.raw_handle() {
            continue;
        }
        let idx = m.index();

        // Padding for gaps
        while next_slot < idx {
            let pad_name = format_ident!("__reserved_{next_slot}");
            vtable_fields.push(quote! {
                #[doc(hidden)]
                pub #pad_name: Option<unsafe extern "C" fn()>
            });
            next_slot += 1;
        }

        let method_name = &m.name;
        let (param_types, fn_ret, _) = vtable_c_fn_sig(m, &handle_names);

        vtable_fields.push(quote! {
            pub #method_name: Option<unsafe extern "C" fn(#(#param_types),*) #fn_ret>
        });
        next_slot = idx + 1;
    }

    // Trailing padding for reserved slots
    if let Some(&max_reserved) = inp.reserved.iter().max() {
        while next_slot <= max_reserved {
            let pad_name = format_ident!("__reserved_{next_slot}");
            vtable_fields.push(quote! {
                #[doc(hidden)]
                pub #pad_name: Option<unsafe extern "C" fn()>
            });
            next_slot += 1;
        }
    }

    // --- Trait impl methods ---
    // Only process own methods (not supertrait methods) for the trait impl.

    // method_sigs are parallel to own non-raw methods (same filtering as @on_library_export).
    let own_methods = &parsed_methods[..inp.own_method_count];
    let own_non_raw: Vec<_> = own_methods.iter().filter(|m| !m.raw_handle()).collect();
    let own_method_impls: Vec<proc_macro2::TokenStream> = own_non_raw
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let method_name = &m.name;

            // Use the pre-baked trait method signature (preserves &mut impl PushStr etc.)
            let sig = &inp.method_sigs[i];

            let (fn_ptr_param_types, fn_ptr_ret, ok_is_handle) = vtable_c_fn_sig(m, &handle_names);
            let fn_ptr_type = quote! {
                unsafe extern "C" fn(#(#fn_ptr_param_types),*) #fn_ptr_ret
            };

            // Build vtable call args (converting Rust values to C)
            let has_impl_trait_params = m.params.iter().any(|p| matches!(&p.kind, crate::meta::MetaParamKind::ImplTrait { .. }));

            let mut vtable_pre = Vec::<proc_macro2::TokenStream>::new();
            let mut vtable_args = Vec::<proc_macro2::TokenStream>::new();
            for p in &m.params {
                let id = &p.name;
                match &p.kind {
                    crate::meta::MetaParamKind::ImplTrait { .. } => {
                        vtable_args.push(quote! { core::ptr::null_mut() });
                    }
                    crate::meta::MetaParamKind::StrSlice => {
                        let vec_id = format_ident!("__{id}_ffierbytes");
                        vtable_pre.push(quote! {
                            let #vec_id: Vec<ffier::FfierBytes> = #id.iter()
                                .map(|s| unsafe { ffier::FfierBytes::from_str(s) })
                                .collect();
                        });
                        vtable_args.push(quote! { #vec_id.as_ptr() });
                        vtable_args.push(quote! { #vec_id.len() });
                    }
                    crate::meta::MetaParamKind::HandleSlice(tp) => {
                        let rt = &tp.rust_type;
                        let vec_id = format_ident!("__{id}_handles");
                        vtable_pre.push(quote! {
                            let #vec_id: Vec<*mut core::ffi::c_void> = #id.iter()
                                .map(|item| <&#rt as FfiType>::into_c(*item))
                                .collect();
                        });
                        vtable_args.push(quote! { #vec_id.as_ptr() });
                        vtable_args.push(quote! { #vec_id.len() });
                    }
                    crate::meta::MetaParamKind::PrimitiveSlice(_) => {
                        // Primitives are layout-compatible with their CRepr,
                        // so just pass the slice pointer and length directly.
                        vtable_args.push(quote! { #id.as_ptr() });
                        vtable_args.push(quote! { #id.len() });
                    }
                    crate::meta::MetaParamKind::Regular(tp) => {
                        let rt = &tp.rust_type;
                        vtable_args.push(quote! { <#rt as FfiType>::into_c(#id) });
                    }
                }
            }

            // Build the call expression
            let raw_call = quote! { {
                #(#vtable_pre)*
                unsafe { __f(self.value.user_data as *mut core::ffi::c_void, #(#vtable_args),*) }
            } };

            // Build vtable_branch (converting C result back to Rust)
            let vtable_branch = if has_impl_trait_params {
                let wrapper_str = wrapper.to_string();
                let name_str = method_name.to_string();
                quote! {
                    let _ = __f;
                    panic!(
                        "{}: vtable dispatch for method `{}` with impl Trait params is not supported",
                        #wrapper_str, #name_str,
                    )
                }
            } else {
                match &m.ret {
                    crate::meta::MetaReturn::Void => raw_call.clone(),
                    crate::meta::MetaReturn::Value(tp) => {
                        let bt = &tp.bridge_type;
                        quote! { unsafe { <#bt as FfiType>::from_c(#raw_call) } }
                    }
                    crate::meta::MetaReturn::HandleSlice { types: tp, .. } => {
                        // HandleSlice vtable dispatch: call __f which returns
                        // FfierObjectArray, reconstruct Vec<&T> from the handle array.
                        let bt = &tp.bridge_type;
                        quote! {{
                            let __arr = #raw_call;
                            if __arr.len == 0 {
                                Vec::new()
                            } else {
                                let __refs: Vec<&#bt> = (0..__arr.len)
                                    .map(|i| unsafe {
                                        let h = ffier::ffier_object_array_get(__arr, i);
                                        ffier::ffier_handle_borrow::<#bt>(h)
                                    })
                                    .collect();
                                unsafe { ffier::ffier_object_array_free(__arr) };
                                __refs
                            }
                        }}
                    }
                    crate::meta::MetaReturn::Result { ok, err_ident, .. } => {
                        let err_ty = format_ident!("{err_ident}");
                        if ok_is_handle {
                            // GLib-style: __f returns handle or null, err through out-param
                            let ok_conversion = match ok {
                                Some(tp) => {
                                    let bt = &tp.bridge_type;
                                    quote! { Ok(unsafe { <#bt as FfiType>::from_c(__raw) }) }
                                }
                                None => quote! { Ok(()) }
                            };
                            quote! {{
                                let mut __err: *mut core::ffi::c_void = core::ptr::null_mut();
                                let __raw = unsafe { __f(
                                    self.value.user_data as *mut core::ffi::c_void,
                                    #(#vtable_args,)*
                                    &mut __err as *mut *mut core::ffi::c_void,
                                ) };
                                if !__raw.is_null() {
                                    #ok_conversion
                                } else {
                                    Err(unsafe { <#err_ty as FfiType>::from_c(__err) })
                                }
                            }}
                        } else {
                            // OutParam: __f returns FfierResult, ok value through out-param
                            let (ok_decl, ok_conversion) = match ok {
                                Some(tp) => {
                                    let bt = &tp.bridge_type;
                                    (
                                        quote! { let mut __out = core::mem::MaybeUninit::<<#bt as FfiType>::CRepr>::uninit(); },
                                        quote! { Ok(unsafe { <#bt as FfiType>::from_c(__out.assume_init()) }) },
                                    )
                                }
                                None => (quote! {}, quote! { Ok(()) })
                            };
                            let out_param = match ok {
                                Some(_) => quote! { __out.as_mut_ptr(), },
                                None => quote! {},
                            };
                            quote! {{
                                #ok_decl
                                let mut __err: *mut core::ffi::c_void = core::ptr::null_mut();
                                let __r = unsafe { __f(
                                    self.value.user_data as *mut core::ffi::c_void,
                                    #(#vtable_args,)*
                                    #out_param
                                    &mut __err as *mut *mut core::ffi::c_void,
                                ) };
                                if __r == ffier::FFIER_RESULT_SUCCESS {
                                    #ok_conversion
                                } else {
                                    Err(unsafe { <#err_ty as FfiType>::from_c(__err) })
                                }
                            }}
                        }
                    }
                }
            };

            // None branch: fallback or panic
            let name_str = method_name.to_string();
            let none_branch = match default_helper_map.get(&name_str) {
                Some(helper_path) => {
                    let param_names: Vec<_> = m.params.iter().map(|p| &p.name).collect();
                    quote! { #helper_path(self #(, #param_names)*) }
                }
                None => {
                    let wrapper_str = wrapper.to_string();
                    quote! {
                        panic!(
                            "{}: required vtable method `{}` not provided",
                            #wrapper_str, #name_str,
                        )
                    }
                }
            };

            // Full method body
            quote! {
                #sig {
                    let __field: Option<#fn_ptr_type> = unsafe {
                        self.value.field_or_none(
                            core::mem::offset_of!(#vtable_struct, #method_name),
                        )
                    };
                    match __field {
                        Some(__f) => { #vtable_branch }
                        None => { #none_branch }
                    }
                }
            }
        })
        .collect();

    // --- Static assertions: verify proc-macro handle detection agrees with FfiType::IS_HANDLE ---
    let handle_assertions: Vec<proc_macro2::TokenStream> = own_non_raw
        .iter()
        .filter_map(|m| {
            if let crate::meta::MetaReturn::Result { ok: Some(tp), .. } = &m.ret {
                let bt = &tp.bridge_type;
                let ok_is_handle = crate::meta::is_result_ok_handle(&m.rust_ret, &handle_names);
                let method_name = m.name.to_string();
                Some(quote! {
                    assert!(
                        <#bt as FfiType>::IS_HANDLE == #ok_is_handle,
                        concat!(
                            "handle detection mismatch for method `",
                            #method_name,
                            "`: proc-macro and FfiType::IS_HANDLE disagree",
                        ),
                    );
                })
            } else {
                None
            }
        })
        .collect();

    let output = quote! {
        #[repr(C)]
        pub struct #vtable_struct {
            pub drop: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
            #(#vtable_fields,)*
        }

        const _: () = {
            use #crate_path::*;

            #(#handle_assertions)*

            impl #trait_path #trait_generics for #wrapper {
                #(#own_method_impls)*
            }

        };
    };

    output.into()
}

/// Automatically re-export ffier metadata alongside a `use` import.
#[proc_macro_attribute]
pub fn reexport(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let use_item = parse_macro_input!(item as ItemUse);

    let mut meta_reexports = Vec::new();
    collect_reexport_paths(&use_item.tree, &mut Vec::new(), &mut meta_reexports);

    let output = quote! {
        #use_item

        #(
            #[doc(hidden)]
            #meta_reexports
        )*
    };

    output.into()
}

/// Walk a `UseTree` and collect `pub use path::__ffier_meta_Type;` statements
/// for each simple (non-glob, non-renamed) leaf.
fn collect_reexport_paths(
    tree: &syn::UseTree,
    prefix: &mut Vec<syn::Ident>,
    out: &mut Vec<proc_macro2::TokenStream>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.clone());
            collect_reexport_paths(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let type_name = &n.ident;
            let meta_alias = format_ident!("__ffier_meta_{type_name}");
            if prefix.is_empty() {
                out.push(quote! { pub use #meta_alias; });
            } else {
                out.push(quote! { pub use #(#prefix)::*::#meta_alias; });
            }
        }
        syn::UseTree::Group(g) => {
            for tree in &g.items {
                collect_reexport_paths(tree, prefix, out);
            }
        }
        syn::UseTree::Rename(_) => {
            // Renamed imports are not supported — the user can manually
            // re-export the metadata if needed.
        }
        syn::UseTree::Glob(_) => {
            // Glob imports can't be handled — we don't know which types exist.
        }
    }
}

// ===========================================================================
// __generate_bridge — proc macro for extern "C" bridge generation
// ===========================================================================

/// Generate `extern "C"` bridge functions and write JSON schema.
///
/// Called by the `__ffier_chain` base case with accumulated metadata
/// from all registered types. Not intended for direct use.
#[doc(hidden)]
#[proc_macro]
pub fn __generate_bridge(input: TokenStream) -> TokenStream {
    bridge::generate_batch_impl(input.into()).into()
}

// ===========================================================================
// generate_bridge — public proc macro entry point
// ===========================================================================

/// Generate `extern "C"` bridge functions for an ffier library.
///
/// Takes a mode keyword, the metadata macro path, and the schema output path.
///
/// - `local` — the macro is in the same crate as `library_definition!`.
///   Uses bare names internally to work around rust#52234.
/// - `external` — the macro is in a different crate (the typical cdylib case).
///   Uses `$crate::` paths for cross-crate resolution.
///
/// `schema_output` is the path where the JSON schema is written.
/// Relative paths are resolved from `CARGO_MANIFEST_DIR` (the directory
/// containing the `Cargo.toml` of the crate calling this macro).
///
/// ```ignore
/// // Same crate as library_definition!:
/// ffier::generate_bridge!(local = __ffier_ft_metadata,
///     schema_output = "../../target/ffier-ft.json");
///
/// // Separate cdylib crate:
/// ffier::generate_bridge!(external = mylib::__ffier_ft_metadata,
///     schema_output = "../../target/ffier-ft.json");
/// ```
#[proc_macro]
pub fn generate_bridge(input: TokenStream) -> TokenStream {
    let input2: proc_macro2::TokenStream = input.into();
    let mut iter = input2.into_iter().peekable();

    // Parse mode keyword
    let mode_ident = match iter.next() {
        Some(proc_macro2::TokenTree::Ident(id)) => id,
        _ => {
            return quote! {
                compile_error!("expected `local = <path>, schema_output = \"...\"` or `external = <path>, schema_output = \"...\"`");
            }
            .into();
        }
    };

    let mode_str = mode_ident.to_string();
    let mode = match mode_str.as_str() {
        "local" => quote! { @local },
        "external" => quote! { @cross_crate },
        _ => {
            return quote! {
                compile_error!(concat!(
                    "unknown mode `", stringify!(#mode_ident),
                    "`: expected `local` or `external`"
                ));
            }
            .into();
        }
    };

    // Consume `=`
    match iter.next() {
        Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {}
        _ => {
            return quote! {
                compile_error!("expected `=` after mode keyword");
            }
            .into();
        }
    }

    // Collect metadata macro path tokens until `,`
    let mut path_tokens = Vec::new();
    loop {
        match iter.peek() {
            Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ',' => {
                iter.next(); // consume `,`
                break;
            }
            Some(_) => path_tokens.push(iter.next().unwrap()),
            None => {
                return quote! {
                    compile_error!("expected `, schema_output = \"...\"` after metadata macro path");
                }
                .into();
            }
        }
    }
    let path: proc_macro2::TokenStream = path_tokens.into_iter().collect();
    if path.is_empty() {
        return quote! {
            compile_error!("expected metadata macro path after `=`");
        }
        .into();
    }

    // Parse `schema_output = "..."`
    match iter.next() {
        Some(proc_macro2::TokenTree::Ident(id)) if id == "schema_output" => {}
        _ => {
            return quote! {
                compile_error!("expected `schema_output = \"...\"`");
            }
            .into();
        }
    }
    match iter.next() {
        Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {}
        _ => {
            return quote! {
                compile_error!("expected `=` after `schema_output`");
            }
            .into();
        }
    }
    let schema_output = match iter.next() {
        Some(proc_macro2::TokenTree::Literal(lit)) => lit,
        _ => {
            return quote! {
                compile_error!("expected string literal for schema_output path");
            }
            .into();
        }
    };

    if iter.next().is_some() {
        return quote! {
            compile_error!("unexpected trailing tokens after schema_output");
        }
        .into();
    }

    quote! { #path!(#mode, #schema_output); }.into()
}
