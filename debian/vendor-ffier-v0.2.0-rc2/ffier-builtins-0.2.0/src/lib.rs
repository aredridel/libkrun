//! Pre-annotated ffier traits for built-in types.
//!
//! This crate defines `PushStr` and `Error` traits with
//! `#[ffier::export]`, generating vtable structs and metadata macros
//! so downstream crates can register them in `library_definition!`:
//!
//! ```ignore
//! ffier::library_definition!("mylib", library_tag = 1,
//!     trait ffier::builtins::PushStr = 10,
//! );
//! ```

pub use ffier_rt::FfierHandle;
pub use ffier_rt::FfierResult;
pub use ffier_rt::ffier_result;

/// Streaming string writer for error messages (and other display output).
///
/// C callers construct a stack-local handle wrapping a callback function.
/// Rust bridge code uses the `fmt::Write` impl to stream `Display` output
/// into the callback without allocating.
///
/// Returns `true` on success, `false` to abort formatting (e.g. buffer
/// full). On `false`, `fmt::Write::write_str` returns `Err(fmt::Error)`
/// which short-circuits `write!()`.
#[ffier_impl::export(bless = "push_str")]
pub trait PushStr {
    #[ffier(index = 0)]
    fn push(&mut self, s: &str) -> bool;
}

/// `fmt::Write` adapter — lets bridge code do `write!(writer, "{}", err)`
/// where `writer` is a `&mut dyn PushStr`.
impl core::fmt::Write for dyn PushStr + '_ {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        if self.push(s) {
            Ok(())
        } else {
            Err(core::fmt::Error)
        }
    }
}

/// Dispatch trait for error types exported across FFI.
///
/// `#[derive(FfiError)]` auto-generates the impl. Self-dispatch generates
/// `ft_error_code(handle)` and `ft_error_message(handle, writer)`.
#[ffier_impl::export(bless = "error_trait")]
pub trait Error {
    #[ffier(index = 0)]
    fn code(&self) -> u32;

    #[ffier(index = 1)]
    fn message(&self, writer: &mut impl PushStr);

    #[ffier(index = 2, raw_handle)]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn result(handle: *const FfierHandle<Self>) -> u64
    where
        Self: Sized,
    {
        let h = unsafe { &*handle };
        ffier_result(h.type_tag, h.value.code())
    }
}
