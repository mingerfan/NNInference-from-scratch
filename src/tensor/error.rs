use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TensorError {
    #[error("tensor shape element count overflows usize: {shape:?}")]
    ShapeOverflow { shape: Vec<usize> },

    #[error(
        "data length does not match shape {shape:?}: expected {expected} elements, got {actual}"
    )]
    DataLengthMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("index rank does not match shape {shape:?}: expected {expected}, got {actual}")]
    RankMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    #[error("index {index} is out of bounds for axis {axis} with size {size}, shape: {shape:?}")]
    IndexOutOfBounds {
        shape: Vec<usize>,
        axis: usize,
        index: usize,
        size: usize,
    },
}

pub type TensorResult<T> = std::result::Result<T, TensorError>;
