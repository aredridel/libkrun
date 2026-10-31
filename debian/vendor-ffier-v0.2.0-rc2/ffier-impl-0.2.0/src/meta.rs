//! Metadata types for ffier's reflection-based architecture.
//!
//! `#[ffier::export]` emits a metadata macro containing structured tokens.
//! Generator proc macros (`generate`) parse
//! these tokens back into the types defined here, then produce code.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, LitBool, LitStr, Token, parse::ParseStream};

// ---------------------------------------------------------------------------
// String case conversion helpers
// ---------------------------------------------------------------------------

pub fn camel_to_snake(s: &str) -> String {
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

pub fn camel_to_upper_snake(s: &str) -> String {
    camel_to_snake(s).to_ascii_uppercase()
}

pub fn snake_to_pascal(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(first) => {
                    let mut s = first.to_uppercase().to_string();
                    s.extend(c);
                    s
                }
                None => String::new(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Peek helper
// ---------------------------------------------------------------------------

/// Peek at the `@tag` identifier from a metadata token stream without consuming it.
pub fn peek_meta_tag(input: &TokenStream) -> String {
    let mut iter = input.clone().into_iter();
    // Skip `@` punct
    if let Some(proc_macro2::TokenTree::Punct(p)) = iter.next()
        && p.as_char() == '@'
        && let Some(proc_macro2::TokenTree::Ident(id)) = iter.next()
    {
        return id.to_string();
    }
    String::new()
}

/// Peek at the type/trait name from a metadata token stream.
///
/// Looks for `name = IDENT` or `trait_name = IDENT` and returns the IDENT.
#[allow(dead_code)]
fn peek_meta_name(input: &TokenStream) -> String {
    let tokens: Vec<proc_macro2::TokenTree> = input.clone().into_iter().collect();
    for i in 0..tokens.len().saturating_sub(2) {
        if let proc_macro2::TokenTree::Ident(ref id) = tokens[i]
            && (id == "name" || id == "trait_name")
            && let proc_macro2::TokenTree::Punct(ref p) = tokens[i + 1]
            && p.as_char() == '='
            && let proc_macro2::TokenTree::Ident(ref name) = tokens[i + 2]
        {
            return name.to_string();
        }
    }
    "Unknown".to_string()
}

/// Peek at a specific `field = VALUE` from a metadata token stream.
/// VALUE can be an ident or a string literal.
pub fn peek_meta_field(input: &TokenStream, field: &str) -> String {
    let tokens: Vec<proc_macro2::TokenTree> = input.clone().into_iter().collect();
    for i in 0..tokens.len().saturating_sub(2) {
        if let proc_macro2::TokenTree::Ident(ref id) = tokens[i]
            && id == field
            && let proc_macro2::TokenTree::Punct(ref p) = tokens[i + 1]
            && p.as_char() == '='
        {
            // The value might be wrapped in a Delimiter::None group
            // (macro_rules! replayed captures produce these).
            let val = match &tokens[i + 2] {
                proc_macro2::TokenTree::Group(g)
                    if g.delimiter() == proc_macro2::Delimiter::None =>
                {
                    g.stream().into_iter().next()
                }
                other => Some(other.clone()),
            };
            match val {
                Some(proc_macro2::TokenTree::Ident(name)) => return name.to_string(),
                Some(proc_macro2::TokenTree::Literal(lit)) => {
                    let s = lit.to_string();
                    return s.trim_matches('"').to_string();
                }
                _ => {}
            }
        }
    }
    "Unknown".to_string()
}

// ---------------------------------------------------------------------------
// Lifetime erasure helpers
// ---------------------------------------------------------------------------

/// Replace all named lifetimes with `'static` in a parsed type.
///
/// Used by annotations to produce types that can be used at the FFI boundary
/// (reexport modules, bridge macros) without free lifetime params.
pub fn erase_lifetimes(ty: &syn::Type) -> syn::Type {
    use syn::visit_mut::VisitMut;
    struct Eraser;
    impl VisitMut for Eraser {
        fn visit_lifetime_mut(&mut self, lt: &mut syn::Lifetime) {
            *lt = syn::Lifetime::new("'static", lt.apostrophe);
        }
    }
    let mut ty = ty.clone();
    Eraser.visit_type_mut(&mut ty);
    ty
}

// ---------------------------------------------------------------------------
// Shared prefix helpers
// ---------------------------------------------------------------------------

/// Return `#[cfg(PRED)]` token stream from a cfg predicate, if present.
pub fn cfg_predicate_attr(cfg: Option<&TokenStream>) -> Option<TokenStream> {
    let pred = cfg?;
    Some(quote! { #[cfg(#pred)] })
}

/// Common prefix formatting for metadata types with a `prefix` field.
pub trait HasPrefix {
    fn prefix(&self) -> &str;

    /// `"{prefix}_"` — C function name prefix.
    fn fn_pfx(&self) -> String {
        format!("{}_", self.prefix())
    }
}

// ---------------------------------------------------------------------------
// Metadata types --- parsed from the metadata macro's token stream
// ---------------------------------------------------------------------------

pub struct MetaExportable {
    pub struct_name: Ident,
    pub struct_path: TokenStream,
    pub prefix: String,
    /// Stable type tag assigned in `library_definition!`. Nonzero when set.
    pub type_tag: u32,
    pub lifetimes: Vec<Ident>,
    pub methods: Vec<MetaMethod>,
    /// `#[cfg(...)]` predicate for the entire type (e.g. `feature = "x"`).
    /// `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaExportable {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

/// Default max dispatch branches before compile error (auto mode).
pub const DEFAULT_MAX_DISPATCH: u64 = 64;

/// How `impl Trait` params are dispatched across concrete types.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Concrete dispatch with default branch limit. Over the limit → compile error.
    Auto,
    /// Force concrete dispatch, no branch limit.
    Concrete,
    /// Force vtable dispatch (requires `#[ffier::export]` on the trait).
    Vtable,
}

/// A bridge/rust type pair used in params and return types.
#[derive(Clone)]
pub struct MetaTypePair {
    pub bridge_type: TokenStream,
    pub rust_type: TokenStream,
    /// When set, the type comes from a foreign ffier library.
    /// The tokens are the crate path (e.g. `other_lib`) whose `FfiType`
    /// and `FfiHandle` traits should be used instead of the local library's.
    pub foreign_crate: Option<TokenStream>,
    /// C typedef name for the foreign handle type (e.g. `"FlForeignConfig"`).
    pub foreign_c_name: Option<String>,
}

pub struct MetaMethod {
    pub name: Ident,
    pub receiver: MetaReceiver,
    /// Doc comment lines, verbatim.
    pub doc: Vec<String>,
    /// Method-level lifetime params (e.g. `[a, b]` from `fn foo<'a, 'b>(...)`).
    pub method_lifetimes: Vec<Ident>,
    pub params: Vec<MetaParam>,
    pub ret: MetaReturn,
    pub rust_ret: TokenStream,
    pub context: MetaMethodContext,
    /// `#[cfg(...)]` predicate tokens (e.g. `feature = "x"`). `None` = unconditional.
    pub cfg: Option<TokenStream>,
}

/// Context-specific fields that are always present together.
pub enum MetaMethodContext {
    /// Method from an exported struct impl or trait impl block.
    Exportable { ffi_name: String, is_builder: bool },
    /// Method from an exported trait definition.
    Trait {
        has_default: bool,
        index: usize,
        raw_handle: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MetaReceiver {
    None,
    Ref,
    Mut,
    Value,
}

impl MetaMethod {
    // --- Exportable context accessors ---

    pub fn ffi_name(&self) -> &str {
        match &self.context {
            MetaMethodContext::Exportable { ffi_name, .. } => ffi_name,
            MetaMethodContext::Trait { .. } => "",
        }
    }

    pub fn is_builder(&self) -> bool {
        matches!(
            &self.context,
            MetaMethodContext::Exportable {
                is_builder: true,
                ..
            }
        )
    }

    pub fn doc(&self) -> &[String] {
        &self.doc
    }

    // --- Trait context accessors ---

    pub fn has_default(&self) -> bool {
        matches!(
            &self.context,
            MetaMethodContext::Trait {
                has_default: true,
                ..
            }
        )
    }

    pub fn index(&self) -> usize {
        match &self.context {
            MetaMethodContext::Trait { index, .. } => *index,
            MetaMethodContext::Exportable { .. } => 0,
        }
    }

    pub fn raw_handle(&self) -> bool {
        matches!(
            &self.context,
            MetaMethodContext::Trait {
                raw_handle: true,
                ..
            }
        )
    }

    // --- Return type accessors ---

    pub fn is_mut(&self) -> bool {
        self.receiver == MetaReceiver::Mut
    }

    /// Return `#[cfg(PRED)]` token stream if this method has a cfg predicate.
    pub fn cfg_attr(&self) -> Option<TokenStream> {
        cfg_predicate_attr(self.cfg.as_ref())
    }
}

pub struct MetaParam {
    pub name: Ident,
    pub kind: MetaParamKind,
}

/// How an `impl Trait` parameter is referenced at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplTraitRefKind {
    /// `param: impl Trait` — by value, handle is consumed.
    Value,
    /// `param: &impl Trait` — shared reference, handle is borrowed.
    Ref,
    /// `param: &mut impl Trait` — mutable reference, handle is borrowed mutably.
    Mut,
}

pub enum MetaParamKind {
    Regular(MetaTypePair),
    StrSlice,
    /// `&[T]` where T is an exported handle type — slice of struct references,
    /// expands to two C params (pointer to handle array + length).
    /// The `MetaTypePair` contains the element type (T, not &[T]).
    HandleSlice(MetaTypePair),
    /// `&[T]` where T is a primitive type (u32, i64, bool, etc.) —
    /// expands to two C params (pointer to element array + length).
    /// The `MetaTypePair` contains the element type (T, not &[T]).
    PrimitiveSlice(MetaTypePair),
    /// `impl Trait` parameter — the generator resolves concrete dispatch
    /// types from the trait map built from `@trait_impl`/`@implementable` metadata entries.
    ImplTrait {
        trait_name: String,
        dispatch: DispatchMode,
        /// How the param is passed: by value, &, or &mut.
        ref_kind: ImplTraitRefKind,
        /// Lifetime arguments on the trait at this usage site (e.g. `["a"]` for `impl Snapshot<'a>`).
        trait_lifetime_args: Vec<Ident>,
    },
}

pub enum MetaReturn {
    Void,
    Value(MetaTypePair),
    Result {
        ok: Option<MetaTypePair>,
        err_ident: String,
    },
    /// `&[&T]` or `&[T]` where T is an exported handle type — returns a
    /// contiguous array of borrowed handles. The `MetaTypePair` contains
    /// the element type T. `direct` is true for `&[T]` (elements are inline),
    /// false for `&[&T]` (elements are references).
    HandleSlice {
        types: MetaTypePair,
        direct: bool,
    },
}

// ---------------------------------------------------------------------------
// Error metadata
// ---------------------------------------------------------------------------

pub struct MetaError {
    pub name: Ident,
    pub path: TokenStream,
    pub prefix: String,
    /// Stable type tag assigned in `library_definition!`. Nonzero when set.
    pub type_tag: u32,
    pub variants: Vec<MetaErrorVariant>,
}

impl HasPrefix for MetaError {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

pub struct MetaErrorVariant {
    pub name: Ident,
    pub code: u32,
    pub message: String,
    /// Type tokens for data-carrying variant fields (e.g. `Box<str>`).
    /// Empty for unit variants.
    pub field_types: Vec<TokenStream>,
}

// ---------------------------------------------------------------------------
// Enum metadata
// ---------------------------------------------------------------------------

pub struct MetaEnum {
    pub name: Ident,
    pub prefix: String,
    /// The `#[repr(...)]` integer type (e.g. "u32", "u64").
    pub repr: String,
    pub variants: Vec<MetaEnumVariant>,
    /// `#[cfg(...)]` predicate for the entire enum. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaEnum {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

pub struct MetaEnumVariant {
    pub name: Ident,
    pub value: u64,
}

// ---------------------------------------------------------------------------
// Bitflags metadata
// ---------------------------------------------------------------------------

/// Bitflags metadata — structurally identical to `MetaEnum` but carries
/// a distinct `@bitflags_constants` tag so generators can emit `bitflags!`
/// invocations (Rust) or annotated `#define` groups (C) instead of plain enums.
pub struct MetaBitflags {
    pub name: Ident,
    pub prefix: String,
    /// The underlying integer type (e.g. "u32", "u64").
    pub repr: String,
    pub variants: Vec<MetaEnumVariant>,
    /// `#[cfg(...)]` predicate for the entire bitflags type. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaBitflags {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ---------------------------------------------------------------------------
// Free function metadata
// ---------------------------------------------------------------------------

pub struct MetaFreeFunction {
    pub name: Ident,
    pub fn_path: TokenStream,
    pub prefix: String,
    pub ffi_name: String,
    pub doc: Vec<String>,
    pub methods: Vec<MetaMethod>,
    /// `#[cfg(...)]` predicate for the entire free function. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaFreeFunction {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ---------------------------------------------------------------------------
// Implementable metadata
// ---------------------------------------------------------------------------

pub struct MetaImplementable {
    pub trait_name: Ident,
    pub trait_path: TokenStream,
    pub prefix: String,
    /// Stable type tag assigned in `library_definition!`. Nonzero when set.
    pub type_tag: u32,
    pub wrapper_name: TokenStream,
    /// Lifetime parameters on the trait definition (e.g. `[a]` for `trait Snapshot<'a>`).
    pub trait_lifetimes: Vec<Ident>,
    pub methods: Vec<MetaMethod>,
    /// Number of methods that belong to this trait (not supertrait methods).
    /// The first `own_method_count` entries in `methods` are this trait's
    /// own methods; the rest are from supertrait `supers(...)` blocks.
    pub own_method_count: usize,
    /// Optional blessing tag for well-known types (e.g. `"error_trait"`).
    pub bless: Option<String>,
    /// Highest vtable slot index (including reserved/retired slots).
    /// Used by code generators to pad the vtable struct up to this slot.
    pub max_vtable_slot: usize,
    /// If true, no vtable wrapper type is generated. C callers cannot
    /// implement this trait — only concrete Rust implementors are dispatched.
    /// Useful for marker traits and traits with supertrait bounds.
    pub no_vtable: bool,
    /// `#[cfg(...)]` predicate for the entire trait. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaImplementable {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ---------------------------------------------------------------------------
// Trait impl metadata (impl Trait for Struct, exported via C ABI)
// ---------------------------------------------------------------------------

pub struct MetaTraitImpl {
    pub trait_name: Ident,
    pub struct_name: Ident,
    pub struct_path: TokenStream,
    pub trait_path: TokenStream,
    pub prefix: String,
    /// Lifetime params from the impl block (e.g. `[a]` from `impl<'a> Trait<'a> for Struct<'a>`).
    pub lifetimes: Vec<Ident>,
    /// Lifetime arguments on the trait (e.g. `["static"]` from `impl Trait<'static> for Struct`,
    /// or `["a"]` from `impl<'a> Trait<'a> for Struct<'a>`). May differ from `lifetimes`.
    pub trait_lifetime_args: Vec<String>,
    /// Lifetime arguments on the struct type (e.g. `["a"]` from `impl<'a> Trait<'a> for View<'a>`,
    /// or `[]` from `impl<'a> Trait<'a> for Widget`). Used to correctly parameterize the struct
    /// in generated impl blocks — only the struct's own lifetimes, not the impl block's.
    pub struct_lifetime_args: Vec<String>,
    pub methods: Vec<MetaMethod>,
    /// `#[cfg(...)]` predicate for the entire trait impl. `None` = unconditional.
    pub cfg_predicate: Option<TokenStream>,
}

impl HasPrefix for MetaTraitImpl {
    fn prefix(&self) -> &str {
        &self.prefix
    }
}

// ---------------------------------------------------------------------------
// Parsing --- from token stream back to metadata types
// ---------------------------------------------------------------------------

// Helper: parse `key = value` where key must match expected
fn expect_key(input: ParseStream, expected: &str) -> syn::Result<()> {
    let key: Ident = input.parse()?;
    if key != expected {
        return Err(syn::Error::new(
            key.span(),
            format!("expected `{expected}`, got `{key}`"),
        ));
    }
    input.parse::<Token![=]>()?;
    Ok(())
}

fn parse_comma(input: ParseStream) -> syn::Result<()> {
    if !input.is_empty() && input.peek(Token![,]) {
        input.parse::<Token![,]>()?;
    }
    Ok(())
}

fn parse_parenthesized_tokens(input: ParseStream) -> syn::Result<TokenStream> {
    let content;
    syn::parenthesized!(content in input);
    content.parse()
}

fn parse_bool(input: ParseStream) -> syn::Result<bool> {
    let lit: LitBool = input.parse()?;
    Ok(lit.value())
}

fn parse_string(input: ParseStream) -> syn::Result<String> {
    let lit: LitStr = input.parse()?;
    Ok(lit.value())
}

/// Parse an optional `cfg_predicate = (predicate tokens),` field from the end
/// of a metadata blob.
fn parse_optional_cfg_predicate(input: ParseStream) -> syn::Result<Option<TokenStream>> {
    if !input.is_empty() && input.peek(Ident) {
        let fork = input.fork();
        if fork.parse::<Ident>().is_ok_and(|id| id == "cfg_predicate") {
            input.parse::<Ident>()?; // cfg_predicate
            input.parse::<Token![=]>()?;
            let tokens = parse_parenthesized_tokens(input)?;
            parse_comma(input)?;
            return Ok(Some(tokens));
        }
    }
    Ok(None)
}

/// Parse a comma-separated list inside the given delimiter.
fn parse_delimited_list<T>(
    content: ParseStream,
    mut parse_item: impl FnMut(ParseStream) -> syn::Result<T>,
) -> syn::Result<Vec<T>> {
    let mut items = Vec::new();
    while !content.is_empty() {
        items.push(parse_item(content)?);
        if !content.is_empty() && content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    Ok(items)
}

/// Parse a `[item, item, ...]` list.
fn parse_bracketed_list<T>(
    input: ParseStream,
    parse_item: impl FnMut(ParseStream) -> syn::Result<T>,
) -> syn::Result<Vec<T>> {
    let content;
    syn::bracketed!(content in input);
    parse_delimited_list(&content, parse_item)
}

/// Parse a `(item, item, ...)` list.
fn parse_parenthesized_list<T>(
    input: ParseStream,
    parse_item: impl FnMut(ParseStream) -> syn::Result<T>,
) -> syn::Result<Vec<T>> {
    let content;
    syn::parenthesized!(content in input);
    parse_delimited_list(&content, parse_item)
}

impl syn::parse::Parse for MetaExportable {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // @exported_impl
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_impl" {
            return Err(syn::Error::new(tag.span(), "expected `exported_impl`"));
        }
        parse_comma(input)?;

        expect_key(input, "name")?;
        let struct_name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "struct_path")?;
        let struct_path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "type_tag")?;
        let type_tag: syn::LitInt = input.parse()?;
        let type_tag = type_tag.base10_parse::<u32>()?;
        parse_comma(input)?;

        expect_key(input, "lifetimes")?;
        let lifetimes = parse_parenthesized_list(input, |inner| inner.parse::<Ident>())?;
        parse_comma(input)?;

        expect_key(input, "methods")?;
        let methods = parse_bracketed_list(input, |inner| inner.parse::<MetaMethod>())?;
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaExportable {
            struct_name,
            struct_path,
            prefix,
            type_tag,
            lifetimes,
            methods,
            cfg_predicate,
        })
    }
}

impl syn::parse::Parse for MetaMethod {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::braced!(content in input);
        let input = &content;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "doc")?;
        let doc = parse_bracketed_list(input, parse_string)?;
        parse_comma(input)?;

        expect_key(input, "receiver")?;
        let receiver = {
            let r: Ident = input.parse()?;
            match r.to_string().as_str() {
                "none" => MetaReceiver::None,
                "r#ref" | "ref" => MetaReceiver::Ref,
                "r#mut" | "mut" => MetaReceiver::Mut,
                "value" => MetaReceiver::Value,
                other => {
                    return Err(syn::Error::new(
                        r.span(),
                        format!("unknown receiver `{other}`"),
                    ));
                }
            }
        };
        parse_comma(input)?;

        // Parse kind-specific fields based on the explicit method_kind tag.
        expect_key(input, "method_kind")?;
        let kind_tag: Ident = input.parse()?;
        parse_comma(input)?;

        let context = match kind_tag.to_string().as_str() {
            "r#impl" | "impl" => {
                expect_key(input, "ffi_name")?;
                let ffi_name = parse_string(input)?;
                parse_comma(input)?;

                expect_key(input, "is_builder")?;
                let is_builder = parse_bool(input)?;
                parse_comma(input)?;

                MetaMethodContext::Exportable {
                    ffi_name,
                    is_builder,
                }
            }
            "definition" => {
                expect_key(input, "has_default")?;
                let has_default = parse_bool(input)?;
                parse_comma(input)?;

                expect_key(input, "index")?;
                let index: syn::LitInt = input.parse()?;
                let index = index.base10_parse::<usize>()?;
                parse_comma(input)?;

                expect_key(input, "raw_handle")?;
                let raw_handle = parse_bool(input)?;
                parse_comma(input)?;

                MetaMethodContext::Trait {
                    has_default,
                    index,
                    raw_handle,
                }
            }
            other => {
                return Err(syn::Error::new(
                    kind_tag.span(),
                    format!("unknown method_kind `{other}`, expected `impl` or `definition`"),
                ));
            }
        };

        expect_key(input, "method_lifetimes")?;
        let method_lifetimes = parse_bracketed_list(input, |inner| inner.parse::<Ident>())?;
        parse_comma(input)?;

        expect_key(input, "params")?;
        let params = parse_bracketed_list(input, |inner| inner.parse::<MetaParam>())?;
        parse_comma(input)?;

        expect_key(input, "ret")?;
        let ret = input.parse::<MetaReturn>()?;
        parse_comma(input)?;

        expect_key(input, "rust_ret")?;
        let rust_ret = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        // Optional: cfg = (predicate tokens),
        let cfg = if !input.is_empty() && input.peek(Ident) {
            let fork = input.fork();
            if fork.parse::<Ident>().is_ok_and(|id| id == "cfg") {
                input.parse::<Ident>()?; // cfg
                input.parse::<Token![=]>()?;
                let tokens = parse_parenthesized_tokens(input)?;
                parse_comma(input)?;
                Some(tokens)
            } else {
                None
            }
        } else {
            None
        };

        Ok(MetaMethod {
            name,
            receiver,
            doc,
            method_lifetimes,
            params,
            ret,
            rust_ret,
            context,
            cfg,
        })
    }
}

/// Parse `bridge_type = (...), rust_type = (...), [foreign_crate = (...),] [foreign_c_name = (...),]` type pair.
fn parse_type_pair(input: ParseStream) -> syn::Result<MetaTypePair> {
    expect_key(input, "bridge_type")?;
    let bridge_type = parse_parenthesized_tokens(input)?;
    parse_comma(input)?;
    expect_key(input, "rust_type")?;
    let rust_type = parse_parenthesized_tokens(input)?;
    parse_comma(input)?;
    // Optional: foreign_crate = (tokens),
    let foreign_crate = if input.peek(Ident)
        && input
            .fork()
            .parse::<Ident>()
            .is_ok_and(|id| id == "foreign_crate")
    {
        expect_key(input, "foreign_crate")?;
        let fc = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;
        Some(fc)
    } else {
        None
    };
    // Optional: foreign_c_name = (literal),
    let foreign_c_name = if input.peek(Ident)
        && input
            .fork()
            .parse::<Ident>()
            .is_ok_and(|id| id == "foreign_c_name")
    {
        expect_key(input, "foreign_c_name")?;
        let content;
        syn::parenthesized!(content in input);
        let lit: syn::LitStr = content.parse()?;
        parse_comma(input)?;
        Some(lit.value())
    } else {
        None
    };
    Ok(MetaTypePair {
        bridge_type,
        rust_type,
        foreign_crate,
        foreign_c_name,
    })
}

impl syn::parse::Parse for MetaParam {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let content;
        syn::braced!(content in input);
        let input = &content;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "kind")?;
        let kind_ident: Ident = input.parse()?;
        let kind = match kind_ident.to_string().as_str() {
            "regular" => {
                parse_comma(input)?;
                let types = parse_type_pair(input)?;
                MetaParamKind::Regular(types)
            }
            "str_slice" => {
                parse_comma(input)?;
                MetaParamKind::StrSlice
            }
            "handle_slice" => {
                parse_comma(input)?;
                let types = parse_type_pair(input)?;
                MetaParamKind::HandleSlice(types)
            }
            "primitive_slice" => {
                parse_comma(input)?;
                let types = parse_type_pair(input)?;
                MetaParamKind::PrimitiveSlice(types)
            }
            "impl_trait" => {
                parse_comma(input)?;
                expect_key(input, "trait_name")?;
                let trait_name = parse_string(input)?;
                parse_comma(input)?;
                expect_key(input, "dispatch")?;
                let dispatch_ident: Ident = input.parse()?;
                let dispatch = match dispatch_ident.to_string().as_str() {
                    "auto" => DispatchMode::Auto,
                    "concrete" => DispatchMode::Concrete,
                    "vtable" => DispatchMode::Vtable,
                    other => {
                        return Err(syn::Error::new(
                            dispatch_ident.span(),
                            format!("unknown dispatch mode `{other}`"),
                        ));
                    }
                };
                parse_comma(input)?;
                expect_key(input, "ref_kind")?;
                let ref_kind_ident: Ident = input.parse()?;
                let ref_kind = match ref_kind_ident.to_string().as_str() {
                    "value" => ImplTraitRefKind::Value,
                    "r#ref" | "ref" => ImplTraitRefKind::Ref,
                    "r#mut" | "mut" => ImplTraitRefKind::Mut,
                    other => {
                        return Err(syn::Error::new(
                            ref_kind_ident.span(),
                            format!("unknown ref_kind `{other}`"),
                        ));
                    }
                };
                parse_comma(input)?;
                expect_key(input, "trait_lifetime_args")?;
                let trait_lifetime_args =
                    parse_bracketed_list(input, |inner| inner.parse::<Ident>())?;
                parse_comma(input)?;
                MetaParamKind::ImplTrait {
                    trait_name,
                    dispatch,
                    ref_kind,
                    trait_lifetime_args,
                }
            }
            other => {
                return Err(syn::Error::new(
                    kind_ident.span(),
                    format!("unknown param kind `{other}`"),
                ));
            }
        };

        Ok(MetaParam { name, kind })
    }
}

impl syn::parse::Parse for MetaReturn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let kind: Ident = input.parse()?;
        match kind.to_string().as_str() {
            "void" => Ok(MetaReturn::Void),
            "value" => {
                let content;
                syn::parenthesized!(content in input);
                let tp = parse_type_pair(&content)?;
                Ok(MetaReturn::Value(tp))
            }
            "result" => {
                let content;
                syn::parenthesized!(content in input);

                expect_key(&content, "ok")?;
                let ok_kind: Ident = content.parse()?;
                let ok = if ok_kind == "void" {
                    None
                } else if ok_kind == "some" {
                    let inner;
                    syn::parenthesized!(inner in content);
                    Some(parse_type_pair(&inner)?)
                } else {
                    return Err(syn::Error::new(ok_kind.span(), "expected `void` or `some`"));
                };
                content.parse::<Token![,]>()?;

                expect_key(&content, "err_ident")?;
                let err_ident = parse_string(&content)?;
                parse_comma(&content)?;

                Ok(MetaReturn::Result { ok, err_ident })
            }
            "handle_slice" | "direct_handle_slice" => {
                let direct = kind == "direct_handle_slice";
                let content;
                syn::parenthesized!(content in input);
                let tp = parse_type_pair(&content)?;
                Ok(MetaReturn::HandleSlice { types: tp, direct })
            }
            other => Err(syn::Error::new(
                kind.span(),
                format!("unknown return kind `{other}`"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Error metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaError {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_error" {
            return Err(syn::Error::new(tag.span(), "expected `exported_error`"));
        }
        parse_comma(input)?;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "path")?;
        let path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "type_tag")?;
        let type_tag: syn::LitInt = input.parse()?;
        let type_tag = type_tag.base10_parse::<u32>()?;
        parse_comma(input)?;

        expect_key(input, "variants")?;
        let variants = parse_bracketed_list(input, |content| {
            let inner;
            syn::braced!(inner in content);
            expect_key(&inner, "name")?;
            let name: Ident = inner.parse()?;
            parse_comma(&inner)?;
            expect_key(&inner, "code")?;
            let code: syn::LitInt = inner.parse()?;
            let code = code.base10_parse::<u32>()?;
            parse_comma(&inner)?;
            expect_key(&inner, "message")?;
            let message = parse_string(&inner)?;
            parse_comma(&inner)?;
            expect_key(&inner, "fields")?;
            let field_types = parse_bracketed_list(&inner, |content| {
                let ts = parse_parenthesized_tokens(content)?;
                parse_comma(content)?;
                Ok(ts)
            })?;
            parse_comma(&inner)?;
            Ok(MetaErrorVariant {
                name,
                code,
                message,
                field_types,
            })
        })?;
        parse_comma(input)?;

        Ok(MetaError {
            name,
            path,
            prefix,
            type_tag,
            variants,
        })
    }
}

// ---------------------------------------------------------------------------
// Enum metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaEnum {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_enum" {
            return Err(syn::Error::new(tag.span(), "expected `exported_enum`"));
        }
        parse_comma(input)?;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "repr")?;
        let repr = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "variants")?;
        let variants = parse_bracketed_list(input, |content| {
            let inner;
            syn::braced!(inner in content);
            expect_key(&inner, "name")?;
            let name: Ident = inner.parse()?;
            parse_comma(&inner)?;
            expect_key(&inner, "value")?;
            let value: syn::LitInt = inner.parse()?;
            let value = value.base10_parse::<u64>()?;
            parse_comma(&inner)?;
            Ok(MetaEnumVariant { name, value })
        })?;
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaEnum {
            name,
            prefix,
            repr,
            variants,
            cfg_predicate,
        })
    }
}

// ---------------------------------------------------------------------------
// Bitflags metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaBitflags {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_bitflags" {
            return Err(syn::Error::new(tag.span(), "expected `exported_bitflags`"));
        }
        parse_comma(input)?;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "repr")?;
        let repr = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "variants")?;
        let variants = parse_bracketed_list(input, |content| {
            let inner;
            syn::braced!(inner in content);
            expect_key(&inner, "name")?;
            let name: Ident = inner.parse()?;
            parse_comma(&inner)?;
            expect_key(&inner, "value")?;
            let value: syn::LitInt = inner.parse()?;
            let value = value.base10_parse::<u64>()?;
            parse_comma(&inner)?;
            Ok(MetaEnumVariant { name, value })
        })?;
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaBitflags {
            name,
            prefix,
            repr,
            variants,
            cfg_predicate,
        })
    }
}

// ---------------------------------------------------------------------------
// Free function metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaFreeFunction {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_fn" {
            return Err(syn::Error::new(tag.span(), "expected `exported_fn`"));
        }
        parse_comma(input)?;

        expect_key(input, "name")?;
        let name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "fn_path")?;
        let fn_path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "ffi_name")?;
        let ffi_name = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "doc")?;
        let doc = parse_bracketed_list(input, parse_string)?;
        parse_comma(input)?;

        expect_key(input, "methods")?;
        let methods = parse_bracketed_list(input, |inner| inner.parse::<MetaMethod>())?;
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaFreeFunction {
            name,
            fn_path,
            prefix,
            ffi_name,
            doc,
            methods,
            cfg_predicate,
        })
    }
}

// ---------------------------------------------------------------------------
// Implementable metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaImplementable {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_trait" {
            return Err(syn::Error::new(tag.span(), "expected `exported_trait`"));
        }
        parse_comma(input)?;

        expect_key(input, "trait_name")?;
        let trait_name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "trait_path")?;
        let trait_path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "type_tag")?;
        let type_tag: syn::LitInt = input.parse()?;
        let type_tag = type_tag.base10_parse::<u32>()?;
        parse_comma(input)?;

        expect_key(input, "wrapper_name")?;
        let wrapper_name = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "trait_lifetimes")?;
        let trait_lifetimes = parse_parenthesized_list(input, |inner| inner.parse::<Ident>())?;
        parse_comma(input)?;

        expect_key(input, "vtable_methods")?;
        let methods = parse_bracketed_list(input, |inner| inner.parse::<MetaMethod>())?;
        parse_comma(input)?;

        expect_key(input, "own_method_count")?;
        let own_method_count: syn::LitInt = input.parse()?;
        let own_method_count = own_method_count.base10_parse::<usize>()?;
        parse_comma(input)?;

        expect_key(input, "max_vtable_slot")?;
        let max_vtable_slot: syn::LitInt = input.parse()?;
        let max_vtable_slot = max_vtable_slot.base10_parse::<usize>()?;
        parse_comma(input)?;

        expect_key(input, "bless")?;
        let bless = if input.peek(syn::LitStr) {
            let lit: syn::LitStr = input.parse()?;
            Some(lit.value())
        } else {
            let ident: syn::Ident = input.parse()?;
            if ident != "none" {
                return Err(syn::Error::new(
                    ident.span(),
                    "expected string literal or `none`",
                ));
            }
            None
        };
        parse_comma(input)?;

        expect_key(input, "no_vtable")?;
        let no_vtable_ident: syn::Ident = input.parse()?;
        let no_vtable = if no_vtable_ident == "yes" {
            true
        } else if no_vtable_ident == "no" {
            false
        } else {
            return Err(syn::Error::new(
                no_vtable_ident.span(),
                "expected `yes` or `no`",
            ));
        };
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaImplementable {
            trait_name,
            trait_path,
            prefix,
            type_tag,
            wrapper_name,
            trait_lifetimes,
            methods,
            own_method_count,
            max_vtable_slot,
            bless,
            no_vtable,
            cfg_predicate,
        })
    }
}

// ---------------------------------------------------------------------------
// Trait impl metadata parsing
// ---------------------------------------------------------------------------

impl syn::parse::Parse for MetaTraitImpl {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        input.parse::<Token![@]>()?;
        let tag: Ident = input.parse()?;
        if tag != "exported_trait_impl" {
            return Err(syn::Error::new(
                tag.span(),
                "expected `exported_trait_impl`",
            ));
        }
        parse_comma(input)?;

        expect_key(input, "trait_name")?;
        let trait_name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "struct_name")?;
        let struct_name: Ident = input.parse()?;
        parse_comma(input)?;

        expect_key(input, "struct_path")?;
        let struct_path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "trait_path")?;
        let trait_path = parse_parenthesized_tokens(input)?;
        parse_comma(input)?;

        expect_key(input, "prefix")?;
        let prefix = parse_string(input)?;
        parse_comma(input)?;

        expect_key(input, "lifetimes")?;
        let lifetimes = parse_parenthesized_list(input, |inner| inner.parse::<Ident>())?;
        parse_comma(input)?;

        expect_key(input, "trait_lifetime_args")?;
        let trait_lifetime_args = parse_bracketed_list(input, parse_string)?;
        parse_comma(input)?;

        expect_key(input, "struct_lifetime_args")?;
        let struct_lifetime_args = parse_bracketed_list(input, parse_string)?;
        parse_comma(input)?;

        expect_key(input, "methods")?;
        let methods = parse_bracketed_list(input, |inner| inner.parse::<MetaMethod>())?;
        parse_comma(input)?;

        let cfg_predicate = parse_optional_cfg_predicate(input)?;

        Ok(MetaTraitImpl {
            trait_name,
            struct_name,
            struct_path,
            trait_path,
            prefix,
            lifetimes,
            trait_lifetime_args,
            struct_lifetime_args,
            methods,
            cfg_predicate,
        })
    }
}

// ---------------------------------------------------------------------------
// Type identity helpers
// ---------------------------------------------------------------------------

/// Extract the last path segment name from a `syn::Type::Path`.
///
/// e.g. `std::result::Result` → `"Result"`, `Widget` → `"Widget"`.
pub fn type_last_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|seg| seg.ident.to_string()),
        _ => None,
    }
}

/// Extract the `Ok` type from `Result<OkType, ErrType>` tokens.
///
/// If the tokens don't parse as a `Result<...>`, returns the input unchanged.
pub fn extract_result_ok_type(tokens: &TokenStream) -> TokenStream {
    if let Ok(ty) = syn::parse2::<syn::Type>(tokens.clone())
        && let syn::Type::Path(tp) = &ty
        && let Some(last) = tp.path.segments.last()
        && last.ident == "Result"
        && let syn::PathArguments::AngleBracketed(args) = &last.arguments
        && let Some(syn::GenericArgument::Type(ok_ty)) = args.args.first()
    {
        return quote! { #ok_ty };
    }
    tokens.clone()
}

/// Check if a `Result<T, E>` return type has an `Ok` type that is a handle,
/// using the original Rust return type tokens (not aliased `bridge_type`).
pub fn is_result_ok_handle(
    rust_ret: &TokenStream,
    handle_names: &std::collections::HashSet<String>,
) -> bool {
    let ok_tokens = extract_result_ok_type(rust_ret);
    let Ok(ok_ty) = syn::parse2::<syn::Type>(ok_tokens) else {
        return false;
    };
    type_last_name(&ok_ty)
        .map(|name| name == "Self" || handle_names.contains(&name))
        .unwrap_or(false)
}
