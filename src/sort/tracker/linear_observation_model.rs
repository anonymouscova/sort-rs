use na::allocator::Allocator;
use na::dimension::DimMin;
use na::dimension::{U4, U7};
use na::OMatrix;
use na::{DefaultAllocator, RealField};
use nalgebra as na;

use adskalman::ObservationModel;

// observation model -------

pub struct PositionObservationModel<R: RealField>
where
    DefaultAllocator: Allocator<R, U4, U4>,
    DefaultAllocator: Allocator<R, U7, U4>,
    DefaultAllocator: Allocator<R, U4, U7>,
    DefaultAllocator: Allocator<R, U7, U7>,
    DefaultAllocator: Allocator<R, U4>,
{
    pub observation_matrix: OMatrix<R, U4, U7>,
    pub observation_matrix_transpose: OMatrix<R, U7, U4>,
    pub observation_noise_covariance: OMatrix<R, U4, U4>,
}

impl<R: RealField> PositionObservationModel<R> {
    #[allow(dead_code)]
    pub fn new() -> Self {
        let one = na::convert(1.0);
        let zero = na::convert(0.0);
        // Create observation model. We only observe the position.
        // Note that from_vec uses row major
        #[rustfmt::skip]
        let observation_matrix = OMatrix::<R,U4,U7>::from_vec(vec![
             one, zero, zero, zero,
            zero,  one, zero, zero,
            zero, zero,  one, zero,
            zero, zero, zero,  one,
            zero, zero, zero, zero,
            zero, zero, zero, zero,
            zero, zero, zero, zero]);

        let ten = na::convert(10.0);
        #[rustfmt::skip]
        let observation_noise_covariance = OMatrix::<R,U4,U4>::new(
             one, zero, zero, zero,
            zero,  one, zero, zero,
            zero, zero,  ten, zero,
            zero, zero, zero,  ten);

        Self {
            observation_matrix,
            observation_matrix_transpose: observation_matrix.transpose(),
            observation_noise_covariance,
        }
    }
}

impl<R: RealField> ObservationModel<R, U7, U4> for PositionObservationModel<R>
where
    DefaultAllocator: Allocator<R, U7, U7>,
    DefaultAllocator: Allocator<R, U4, U7>,
    DefaultAllocator: Allocator<R, U7, U4>,
    DefaultAllocator: Allocator<R, U4, U4>,
    DefaultAllocator: Allocator<R, U7>,
    DefaultAllocator: Allocator<R, U4>,
    DefaultAllocator: Allocator<(usize, usize), U4>,
    U4: DimMin<U4, Output = U4>,
{
    fn H(&self) -> &OMatrix<R, U4, U7> {
        &self.observation_matrix
    }
    fn HT(&self) -> &OMatrix<R, U7, U4> {
        &self.observation_matrix_transpose
    }
    fn R(&self) -> &OMatrix<R, U4, U4> {
        &self.observation_noise_covariance
    }
}
