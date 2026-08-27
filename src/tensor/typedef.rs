use super::{TensorError, TensorResult};

pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn checked_numel(shape: &[usize]) -> TensorResult<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }

    shape
        .iter()
        .try_fold(1usize, |numel, dim| numel.checked_mul(*dim))
        .ok_or_else(|| TensorError::ShapeOverflow {
            shape: shape.to_vec(),
        })
}

// 在Chapter1.1中，我们暂时先使用Row-major进行布局
impl Tensor {
    pub fn new_with_value(shape: &[usize], value: f32) -> TensorResult<Self> {
        // numel = number of elements
        let numel = checked_numel(shape)?;
        Ok(Tensor {
            data: vec![value; numel],
            shape: shape.to_vec(),
        })
    }

    pub fn zeros(shape: &[usize]) -> TensorResult<Self> {
        Self::new_with_value(shape, 0f32)
    }

    pub fn ones(shape: &[usize]) -> TensorResult<Self> {
        Self::new_with_value(shape, 1f32)
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> TensorResult<Self> {
        let numel = checked_numel(shape)?;
        if numel != data.len() {
            Err(TensorError::DataLengthMismatch {
                shape: shape.to_vec(),
                expected: numel,
                actual: data.len(),
            })
        } else {
            Ok(Tensor {
                data,
                shape: shape.to_vec(),
            })
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn offset(&self, indices: &[usize]) -> TensorResult<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                shape: self.shape.clone(),
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }

        for (axis, (&index, &size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if index >= size {
                return Err(TensorError::IndexOutOfBounds {
                    shape: self.shape.clone(),
                    axis,
                    index,
                    size,
                });
            }
        }

        let mut offset = 0;
        for (&index, &size) in indices.iter().zip(self.shape.iter()) {
            offset = offset * size + index;
        }

        debug_assert!(offset < self.data.len());
        Ok(offset)
    }

    pub fn get(&self, indices: &[usize]) -> TensorResult<f32> {
        let offset = self.offset(indices)?;
        Ok(self.data[offset])
    }

    pub fn set(&mut self, indices: &[usize], value: f32) -> TensorResult<()> {
        let offset = self.offset(indices)?;
        self.data[offset] = value;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, TensorError};

    #[test]
    fn indexes_three_dimensional_tensor_in_row_major_order() {
        let data = (0..24).map(|value| value as f32).collect();
        let mut tensor = Tensor::from_vec(data, &[2, 3, 4]).unwrap();

        assert_eq!(tensor.get(&[1, 0, 0]), Ok(12.0));
        assert_eq!(tensor.get(&[1, 2, 3]), Ok(23.0));

        tensor.set(&[1, 2, 3], 42.0).unwrap();
        assert_eq!(tensor.get(&[1, 2, 3]), Ok(42.0));
    }

    #[test]
    fn rejects_an_index_that_is_out_of_bounds_on_any_axis() {
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let expected_error = TensorError::IndexOutOfBounds {
            shape: vec![2, 3],
            axis: 1,
            index: 3,
            size: 3,
        };

        assert_eq!(tensor.get(&[0, 3]), Err(expected_error.clone()));
        assert_eq!(tensor.set(&[0, 3], 42.0), Err(expected_error));
        assert_eq!(tensor.get(&[1, 0]), Ok(4.0));
    }

    #[test]
    fn handles_zero_dimensions_and_shape_overflow() {
        assert_eq!(Tensor::zeros(&[]).unwrap().numel(), 1);
        let empty = Tensor::zeros(&[usize::MAX, 2, 0]).unwrap();
        assert_eq!(empty.numel(), 0);
        assert!(matches!(
            empty.get(&[usize::MAX - 1, 1, 0]),
            Err(TensorError::IndexOutOfBounds { axis: 2, .. })
        ));
        assert!(matches!(
            Tensor::from_vec(Vec::new(), &[usize::MAX, 2]),
            Err(TensorError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn reports_data_length_and_rank_mismatches() {
        assert!(matches!(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[2, 2]),
            Err(TensorError::DataLengthMismatch {
                expected: 4,
                actual: 3,
                ..
            })
        ));

        let tensor = Tensor::zeros(&[2, 3]).unwrap();
        assert_eq!(
            tensor.get(&[1]),
            Err(TensorError::RankMismatch {
                shape: vec![2, 3],
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn constructs_filled_tensors_and_reports_metadata() {
        let zeros = Tensor::zeros(&[2, 3]).unwrap();
        assert_eq!(zeros.shape(), &[2, 3]);
        assert_eq!(zeros.ndim(), 2);
        assert_eq!(zeros.numel(), 6);

        for row in 0..2 {
            for column in 0..3 {
                assert_eq!(zeros.get(&[row, column]), Ok(0.0));
            }
        }

        let ones = Tensor::ones(&[2, 2]).unwrap();
        for row in 0..2 {
            for column in 0..2 {
                assert_eq!(ones.get(&[row, column]), Ok(1.0));
            }
        }
    }

    #[test]
    fn reads_and_writes_scalar_tensor() {
        let mut scalar = Tensor::from_vec(vec![3.0], &[]).unwrap();

        assert_eq!(scalar.shape(), &[]);
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.numel(), 1);
        assert_eq!(scalar.get(&[]), Ok(3.0));

        scalar.set(&[], 4.0).unwrap();
        assert_eq!(scalar.get(&[]), Ok(4.0));
    }
}
