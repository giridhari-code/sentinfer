use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    /// Buffer lengths or parameters do not agree.
    Shape(String),
    /// An index read from an input buffer is out of range.
    Index(String),
    /// The backend does not implement this op / dtype / layout.
    Unsupported(String),
    /// Backend-specific failure (allocation, launch, copy).
    Backend(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "shape error: {m}"),
            Self::Index(m) => write!(f, "index error: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

impl std::error::Error for KernelError {}

pub type Result<T> = std::result::Result<T, KernelError>;

macro_rules! ensure {
    ($cond:expr, $kind:path, $($arg:tt)+) => {
        if !($cond) {
            return Err($kind(format!($($arg)+)));
        }
    };
}
pub(crate) use ensure;
