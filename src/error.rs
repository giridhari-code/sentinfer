use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SentinelError {
    Config(String),
    InvalidState(String),
    BlockManager(String),
    /// Feature present in the C++ engine but not ported yet.
    Unsupported(String),
    /// File or OS error (checkpoints, datasets).
    Io(String),
}

impl fmt::Display for SentinelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "config error: {m}"),
            Self::InvalidState(m) => write!(f, "invalid state: {m}"),
            Self::BlockManager(m) => write!(f, "block manager: {m}"),
            Self::Unsupported(m) => write!(f, "not supported yet: {m}"),
            Self::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for SentinelError {}

pub type Result<T> = std::result::Result<T, SentinelError>;

/// Early-return `Err(kind(format!(..)))` when `cond` is false.
/// The message is only formatted on failure, so it costs nothing on the hot path.
macro_rules! ensure {
    ($cond:expr, $kind:path, $($arg:tt)+) => {
        if !($cond) {
            return Err($kind(format!($($arg)+)));
        }
    };
}
pub(crate) use ensure;
