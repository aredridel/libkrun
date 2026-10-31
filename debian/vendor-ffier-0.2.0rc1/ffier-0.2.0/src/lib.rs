pub use ffier_impl::*;
pub use ffier_rt::*;

// Re-export built-in traits at the crate root so `ffier::Error`,
// `ffier::PushStr` work in user code and generated derive output.
pub use ffier_builtins::Error;
pub use ffier_builtins::PushStr;

/// Pre-annotated built-in traits (PushStr, etc.) ready for use in
/// `library_definition!` without redeclaring the trait signature.
pub mod builtins {
    pub use ffier_builtins::*;
}
