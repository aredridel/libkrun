use std::ffi::CStr;

use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Handle layout — heap-allocated box with RTTI header
// ---------------------------------------------------------------------------

/// The handle struct wrapping every value exported across FFI.
///
/// Always heap-allocated via `Box<FfierHandle<T>>`. The FFI pointer is
/// `*mut c_void` pointing to the `FfierHandle<T>`. All handle types —
/// regular exported types, error types, vtable types — share this
/// exact layout, differing only in `T`.
#[repr(C)]
pub struct FfierHandle<T> {
    pub type_tag: u32,
    pub metadata: u32,
    pub value: T,
}

/// A borrowed handle: same 8-byte header as `FfierHandle<T>`, but the
/// value field is a raw pointer back to the real `T` instead of an inline
/// copy. Fixed 16 bytes regardless of T size.
///
/// Allocated by `ffier_handle_new_borrowed`. The runtime functions
/// (`ffier_handle_borrow`, `ffier_handle_drop`) check `METADATA_OWNED`
/// to decide whether the handle is owned (value inline in `FfierHandle<T>`)
/// or borrowed (value is a pointer in `FfierBorrowedHandle`).
#[repr(C)]
pub struct FfierBorrowedHandle {
    pub type_tag: u32,
    pub metadata: u32,
    /// Raw pointer to the real value (lives in the parent struct).
    pub ptr: *const c_void,
}

/// Metadata bit 0: the handle was Box-allocated by ffier and ffier owns the
/// inner value. When set, `ffier_handle_drop` runs `T::drop` and deallocates.
/// When clear, ffier will NOT run the inner destructor.
pub const METADATA_OWNED: u32 = 1;

/// Metadata bit 1: the handle is a `FfierBorrowedHandle` — the value
/// field is a `*const c_void` pointer to the real T, not an inline T.
/// Always set by `ffier_handle_new_borrowed`. Never set for C-side
/// stack-allocated vtable handles or owned handles.
pub const METADATA_BORROWED: u32 = 2;

/// Metadata bit 2: the handle is an element of a contiguous
/// `FfierObjectArray` allocation. Individual destroy is invalid —
/// the whole array must be freed via the array's destroy function.
pub const METADATA_ARRAY_ELEMENT: u32 = 4;

/// Metadata bit 3: the handle carries default-dispatch re-entry metadata
/// for an exported trait method.
///
/// When set, the upper bits encode the trait method index. This is used by
/// generated Rust client bindings to mark temporary vtable handles created
/// for defaulted methods so the bridge can call the library default helper
/// instead of recursing back through the vtable.
pub const METADATA_DEFAULT_DISPATCH: u32 = 8;

/// Bit shift for the method index encoded alongside
/// [`METADATA_DEFAULT_DISPATCH`].
pub const METADATA_DEFAULT_DISPATCH_SHIFT: u32 = 4;

/// Encode metadata for a temporary handle used by default-method self-dispatch.
#[inline]
pub fn default_dispatch_metadata(method_index: usize) -> u32 {
    let method_index = u32::try_from(method_index).expect("method index does not fit in u32");
    METADATA_DEFAULT_DISPATCH | (method_index << METADATA_DEFAULT_DISPATCH_SHIFT)
}

/// Decode the method index from default-dispatch metadata.
///
/// Returns `None` when the handle metadata does not carry the
/// [`METADATA_DEFAULT_DISPATCH`] flag.
#[inline]
pub fn default_dispatch_method_index(metadata: u32) -> Option<u32> {
    if metadata & METADATA_DEFAULT_DISPATCH == 0 {
        None
    } else {
        Some(metadata >> METADATA_DEFAULT_DISPATCH_SHIFT)
    }
}

// ---------------------------------------------------------------------------
// Handle introspection
// ---------------------------------------------------------------------------

/// Read the type tag from a raw handle pointer (offset 0).
///
/// # Safety
/// `handle` must point to a valid `FfierHandle<T>`.
#[inline]
pub unsafe fn handle_type_tag(handle: *const c_void) -> u32 {
    unsafe { *(handle as *const u32) }
}

/// Read the metadata field from a raw handle pointer (offset 4).
///
/// # Safety
/// `handle` must point to a valid `FfierHandle<T>`.
#[inline]
pub unsafe fn handle_metadata(handle: *const c_void) -> u32 {
    unsafe { *((handle as *const u32).add(1)) }
}

// ---------------------------------------------------------------------------
// Handle allocation
// ---------------------------------------------------------------------------

/// Allocate a new owned handle on the heap and return a raw pointer.
///
/// Sets [`METADATA_OWNED`] — `ffier_handle_drop` will run `T::drop` and
/// deallocate. The returned pointer must eventually be passed to
/// `ffier_handle_drop` or `ffier_handle_consume` to avoid leaking.
#[inline]
pub fn ffier_handle_new<T>(tag: u32, value: T) -> *mut c_void {
    debug_assert!(tag != 0, "type tag must be nonzero");
    let handle = Box::new(FfierHandle {
        type_tag: tag,
        metadata: METADATA_OWNED,
        value,
    });
    Box::into_raw(handle) as *mut c_void
}

/// Allocate a new owned handle with custom metadata (e.g. for vtable handles).
///
/// `METADATA_OWNED` is OR'd in automatically — callers should not include it.
#[inline]
pub fn ffier_handle_new_with_metadata<T>(tag: u32, metadata: u32, value: T) -> *mut c_void {
    debug_assert!(tag != 0, "type tag must be nonzero");
    let handle = Box::new(FfierHandle {
        type_tag: tag,
        metadata: metadata | METADATA_OWNED,
        value,
    });
    Box::into_raw(handle) as *mut c_void
}

/// Allocate a borrowed handle: stores a raw pointer to the real `T` in
/// a 16-byte `FfierBorrowedHandle` (same header layout as `FfierHandle`).
///
/// The returned handle has a valid type_tag so it can be passed to methods
/// that type-check the handle. The value is NOT copied — the pointer is
/// stored directly. On destroy, ffier deallocates the 16-byte shell
/// without touching the pointed-to value.
///
/// # Safety
/// - `value` must point to a valid, aligned `T`.
/// - The caller must ensure the source `T` outlives all uses of this handle.
#[inline]
pub unsafe fn ffier_handle_new_borrowed<T>(tag: u32, value: *const T) -> *mut c_void {
    debug_assert!(tag != 0, "type tag must be nonzero");
    let handle = Box::new(FfierBorrowedHandle {
        type_tag: tag,
        metadata: METADATA_BORROWED, // not owned, layout is pointer
        ptr: value as *const c_void,
    });
    Box::into_raw(handle) as *mut c_void
}

// ---------------------------------------------------------------------------
// Handle borrow
// ---------------------------------------------------------------------------

/// Borrow a shared reference to the value inside a handle.
///
/// For owned handles (`METADATA_OWNED` set), the value is inline in the
/// `FfierHandle<T>`. For borrowed handles, the value field is a pointer
/// in a `FfierBorrowedHandle` that is followed to reach the real `T`.
///
/// # Safety
/// - `handle` must point to a valid `FfierHandle<T>` or `FfierBorrowedHandle`.
/// - The handle must be alive for the lifetime of the returned reference.
#[inline]
pub unsafe fn ffier_handle_borrow<T>(handle: *const c_void) -> &'static T {
    debug_assert!(!handle.is_null(), "null handle");
    debug_assert!(
        unsafe { handle_type_tag(handle) } != 0,
        "uninitialized handle"
    );
    if unsafe { handle_metadata(handle) } & METADATA_BORROWED != 0 {
        let bh = unsafe { &*(handle as *const FfierBorrowedHandle) };
        unsafe { &*(bh.ptr as *const T) }
    } else {
        unsafe { &(*(handle as *const FfierHandle<T>)).value }
    }
}

/// Borrow a mutable reference to the value inside a handle.
///
/// For owned handles, returns a reference to the inline value. For
/// borrowed handles, follows the stored pointer.
///
/// # Safety
/// - `handle` must point to a valid `FfierHandle<T>` or `FfierBorrowedHandle`.
/// - The caller must have exclusive access.
#[inline]
pub unsafe fn ffier_handle_borrow_mut<T>(handle: *mut c_void) -> &'static mut T {
    debug_assert!(!handle.is_null(), "null handle");
    debug_assert!(
        unsafe { handle_type_tag(handle) } != 0,
        "uninitialized handle"
    );
    if unsafe { handle_metadata(handle) } & METADATA_BORROWED != 0 {
        let bh = unsafe { &*(handle as *const FfierBorrowedHandle) };
        unsafe { &mut *(bh.ptr as *mut T) }
    } else {
        unsafe { &mut (*(handle as *mut FfierHandle<T>)).value }
    }
}

// ---------------------------------------------------------------------------
// Handle consume — take ownership, free the box
// ---------------------------------------------------------------------------

/// Consume a handle: move the value out and free the allocation.
///
/// Only valid for **owned** handles. Panics (in debug) on borrowed handles.
///
/// # Safety
/// - `handle` must point to a valid, heap-allocated `FfierHandle<T>`.
/// - After this call the handle pointer is dangling.
#[inline]
pub unsafe fn ffier_handle_consume<T>(handle: *mut c_void) -> T {
    debug_assert!(!handle.is_null(), "consuming null handle");
    debug_assert!(
        unsafe { handle_type_tag(handle) } != 0,
        "consuming uninitialized handle"
    );
    debug_assert!(
        unsafe { handle_metadata(handle) } & METADATA_BORROWED == 0,
        "cannot consume a borrowed handle"
    );
    let boxed = unsafe { Box::from_raw(handle as *mut FfierHandle<T>) };
    boxed.value
}

// ---------------------------------------------------------------------------
// Handle drop — drop the value and free the allocation
// ---------------------------------------------------------------------------

/// Drop a handle and free its allocation.
///
/// - **Owned** (`METADATA_OWNED` set): runs `T::drop` and frees the
///   `Box<FfierHandle<T>>`.
/// - **Borrowed** (`METADATA_OWNED` clear): frees the 16-byte
///   `Box<FfierBorrowedHandle>` shell. The pointed-to value is not
///   touched — the real owner is responsible for its lifetime.
///
/// # Safety
/// - `handle` must point to a valid, Box-allocated `FfierHandle<T>` or
///   `FfierBorrowedHandle`, or be null (no-op).
#[inline]
pub unsafe fn ffier_handle_drop<T>(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    debug_assert!(
        unsafe { handle_type_tag(handle) } != 0,
        "dropping uninitialized handle"
    );
    let meta = unsafe { handle_metadata(handle) };
    assert!(
        meta & METADATA_ARRAY_ELEMENT == 0,
        "cannot destroy an individual array element — free the whole array instead"
    );
    if meta & METADATA_BORROWED != 0 {
        // Borrowed: dealloc the 16-byte pointer shell only.
        drop(unsafe { Box::from_raw(handle as *mut FfierBorrowedHandle) });
    } else if meta & METADATA_OWNED != 0 {
        // Owned: drop inner value + dealloc.
        drop(unsafe { Box::from_raw(handle as *mut FfierHandle<T>) });
    }
    // else: not owned, not borrowed (e.g. C stack vtable handle) — do nothing.
}

// ---------------------------------------------------------------------------
// VtableHandle --- payload for exported trait handles
// ---------------------------------------------------------------------------

/// Vtable handle payload — stored inline in the handle's payload region
/// for exported trait handles.
#[repr(C)]
pub struct VtableHandle {
    pub vtable_ptr: *const c_void,
    pub user_data: *const c_void,
    /// Size of the vtable struct as provided by the caller (truncated to
    /// `u16`; max 65535 bytes — more than enough for any vtable). Used for
    /// forward/backward compatibility: fields at offsets beyond this size
    /// are treated as `None` (default dispatch). This allows older clients
    /// (smaller vtable) to work with newer libraries (larger vtable) and
    /// vice versa.
    pub vtable_size: u16,
}

impl VtableHandle {
    /// Read an `Option<extern "C" fn(...)>` vtable field with bounds checking.
    ///
    /// Returns `None` if the field extends beyond `self.vtable_size`,
    /// providing forward/backward-compatible default dispatch for fields
    /// not present in the vtable.
    ///
    /// # Safety
    /// - `vtable_ptr` must point to a valid vtable struct of at least
    ///   `self.vtable_size` bytes.
    /// - `field_offset` must be the correct offset of an `Option<F>` field.
    #[inline]
    pub unsafe fn field_or_none<F: Copy>(&self, field_offset: usize) -> Option<F> {
        if field_offset + core::mem::size_of::<Option<F>>() > self.vtable_size as usize {
            None
        } else {
            unsafe { *(self.vtable_ptr.byte_add(field_offset) as *const Option<F>) }
        }
    }
}

// ---------------------------------------------------------------------------
// FfierObjectArray --- contiguous array of borrowed handles
// ---------------------------------------------------------------------------

/// `#[repr(C)]` array of borrowed handles returned from methods that
/// produce `&[&T]` or `&[T]`. The backing storage is a single contiguous
/// `Box<[FfierBorrowedHandle]>` allocation, leaked to C as an opaque
/// pointer plus length.
///
/// Individual elements must NOT be passed to `destroy` — the whole array
/// is freed in one call via the array's destroy function, which
/// reconstructs and drops the `Box<[FfierBorrowedHandle]>`.
///
/// Access elements via `ffier_object_array_get()`, not direct indexing.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FfierObjectArray {
    /// Opaque pointer to the first `FfierBorrowedHandle` element.
    /// Null when `len == 0`. NOT directly indexable from C —
    /// use `ffier_object_array_get()`.
    _opaque: *const FfierBorrowedHandle,
    pub len: usize,
}

impl FfierObjectArray {
    pub const EMPTY: Self = Self {
        _opaque: core::ptr::null(),
        len: 0,
    };

    /// Create from a leaked `Box<[FfierBorrowedHandle]>`.
    pub fn from_raw(ptr: *const FfierBorrowedHandle, len: usize) -> Self {
        Self { _opaque: ptr, len }
    }
}

/// Get a handle pointer to the i-th element of a `FfierObjectArray`.
///
/// Returns a `*mut c_void` pointing to the `FfierBorrowedHandle` at
/// index `i`. The handle has valid RTTI headers and can be passed to
/// methods that type-check the handle.
///
/// # Safety
/// - `arr` must be a valid `FfierObjectArray`.
/// - `i` must be `< arr.len`.
#[inline]
pub unsafe fn ffier_object_array_get(arr: FfierObjectArray, i: usize) -> *mut c_void {
    debug_assert!(i < arr.len, "index out of bounds");
    unsafe { arr._opaque.add(i) as *mut c_void }
}

/// Free a `FfierObjectArray` by reconstructing and dropping the
/// `Box<[FfierBorrowedHandle]>`.
///
/// # Safety
/// - `arr` must have been created via `FfierObjectArray::from_raw` (or be EMPTY).
/// - Must only be called once per array.
#[inline]
pub unsafe fn ffier_object_array_free(arr: FfierObjectArray) {
    if !arr._opaque.is_null() && arr.len > 0 {
        let raw_slice =
            core::ptr::slice_from_raw_parts(arr._opaque, arr.len) as *mut [FfierBorrowedHandle];
        drop(unsafe { Box::from_raw(raw_slice) });
    }
}

// ---------------------------------------------------------------------------
// FfierBytes --- zero-copy byte slice for C FFI (&[u8], &str, &Path)
// ---------------------------------------------------------------------------

/// `#[repr(C)]` byte slice passed across FFI. In C, each usage gets a
/// typedef (`Str`, `Bytes`, `Path`) from the same underlying struct.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FfierBytes {
    pub data: *const u8,
    pub len: usize,
}

impl FfierBytes {
    pub const EMPTY: Self = Self {
        data: core::ptr::null(),
        len: 0,
    };

    /// # Safety
    /// `data` must be valid for `len` bytes, or null (returns `&[]`).
    pub unsafe fn as_bytes(&self) -> &[u8] {
        if self.data.is_null() {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.data, self.len) }
        }
    }

    /// # Safety
    /// `data` must point to valid UTF-8 of length `len`.
    pub unsafe fn as_str_unchecked(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(self.as_bytes()) }
    }

    /// # Safety
    /// `data` must point to valid UTF-8 of length `len`.
    #[cfg(unix)]
    pub unsafe fn as_path(&self) -> &std::path::Path {
        use std::os::unix::ffi::OsStrExt;
        std::path::Path::new(std::ffi::OsStr::from_bytes(unsafe { self.as_bytes() }))
    }

    /// Create from a byte slice. The returned `FfierBytes` holds a raw pointer
    /// into `b`'s data — no copy is made.
    ///
    /// # Safety
    /// The caller must ensure the source data outlives the `FfierBytes` and any
    /// FFI call that receives it.
    pub unsafe fn from_bytes(b: &[u8]) -> Self {
        Self {
            data: b.as_ptr(),
            len: b.len(),
        }
    }

    /// Create from a string slice. The returned `FfierBytes` holds a raw pointer
    /// into `s`'s data — no copy is made.
    ///
    /// # Safety
    /// The caller must ensure the source string outlives the `FfierBytes` and any
    /// FFI call that receives it.
    pub unsafe fn from_str(s: &str) -> Self {
        unsafe { Self::from_bytes(s.as_bytes()) }
    }

    /// Create from a path. The returned `FfierBytes` holds a raw pointer
    /// into `p`'s data — no copy is made.
    ///
    /// # Safety
    /// The caller must ensure the source path outlives the `FfierBytes` and any
    /// FFI call that receives it.
    #[cfg(unix)]
    pub unsafe fn from_path(p: &std::path::Path) -> Self {
        use std::os::unix::ffi::OsStrExt;
        unsafe { Self::from_bytes(p.as_os_str().as_bytes()) }
    }
}

// ---------------------------------------------------------------------------
// FfiError --- trait for error enums exported via FFI
// ---------------------------------------------------------------------------

/// Trait for error enums exported across the FFI boundary.
///
/// Requires `std::error::Error` (which implies `Display`). The `Display`
/// impl is used for error messages streamed on demand via `WriteStr`.
pub trait FfiError: std::error::Error + Sized {
    /// Variant code (lower 32 bits of `FfierResult`).
    fn code(&self) -> u32;

    /// Static human-readable message for a variant code (no allocation).
    fn static_message(code: u32) -> &'static CStr;

    /// `(CONSTANT_NAME, value)` pairs for C `#define` generation.
    fn codes() -> &'static [(&'static str, u32)];

    /// Shallow-copy the variant's payload CRepr into `out_buf`.
    /// The written CRepr borrows from `&self` — the caller must not
    /// outlive the error value. `buf_size` is checked for safety.
    /// Fieldless (empty-tuple) variants are a no-op.
    ///
    /// # Safety
    /// `out_buf` must point to a valid, aligned buffer of at least
    /// `buf_size` bytes.
    unsafe fn payload(&self, _out_buf: *mut core::ffi::c_void, _buf_size: usize) {}
}

// ---------------------------------------------------------------------------
// FfierResult --- packed u64 error code (upper 32 = type tag, lower 32 = code)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Library tag + type tag composition
// ---------------------------------------------------------------------------

/// Number of bits reserved for the library tag in the upper part of `type_tag`.
pub const LIBRARY_TAG_BITS: u32 = 8;

/// Number of bits available for the per-type tag.
pub const TYPE_TAG_BITS: u32 = 32 - LIBRARY_TAG_BITS; // 24

/// Maximum valid library tag value (2^8 - 1 = 255).
pub const MAX_LIBRARY_TAG: u32 = (1 << LIBRARY_TAG_BITS) - 1;

/// Maximum valid per-type tag value (2^24 - 1 = 16_777_215).
pub const MAX_TYPE_TAG: u32 = (1 << TYPE_TAG_BITS) - 1;

/// Compose a full `type_tag` from a library tag and a per-type tag.
///
/// Layout: `[library_tag: 8 bits | type_tag: 24 bits]`
#[inline]
pub const fn compose_tag(library_tag: u32, type_tag: u32) -> u32 {
    (library_tag << TYPE_TAG_BITS) | type_tag
}

/// Extract the library tag (upper 8 bits) from a composed type tag.
#[inline]
pub const fn extract_library_tag(tag: u32) -> u32 {
    tag >> TYPE_TAG_BITS
}

/// Extract the per-type tag (lower 24 bits) from a composed type tag.
#[inline]
pub const fn extract_type_tag(tag: u32) -> u32 {
    tag & MAX_TYPE_TAG
}

// ---------------------------------------------------------------------------
// FfierResult --- packed u64 error code (upper 32 = type tag, lower 32 = code)
// ---------------------------------------------------------------------------

/// Packed error result: `0` = success, nonzero = `(type_tag << 32) | code`.
///
/// Users compare against generated constants (`FT_ERROR_CALC_OVERFLOW` etc.).
/// The internal layout is an implementation detail.
pub type FfierResult = u64;

/// Build a `FfierResult` from a type tag and variant code.
#[inline]
pub fn ffier_result(type_tag: u32, code: u32) -> FfierResult {
    ((type_tag as u64) << 32) | (code as u64)
}

/// Extract the type tag from a `FfierResult` (upper 32 bits).
#[inline]
pub fn ffier_result_type_tag(r: FfierResult) -> u32 {
    (r >> 32) as u32
}

/// Extract the error code from a `FfierResult` (lower 32 bits).
#[inline]
pub fn ffier_result_code(r: FfierResult) -> u32 {
    r as u32
}

/// Const-compatible string equality check.
pub const fn const_str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Success value.
pub const FFIER_RESULT_SUCCESS: FfierResult = 0;
