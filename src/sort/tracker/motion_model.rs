use adskalman::TransitionModelLinearNoControl;
use na::allocator::Allocator;
use na::dimension::{U4, U7};
use na::OMatrix;
use na::{DefaultAllocator, RealField};
use nalgebra as na;

/// Largely based on
/// - https://github.com/abewley/sort
/// - https://github.com/strawlab/adskalman-rs
///
/// # Model
/// x = [ u, v, s, r, v_u, v_v, v_s ]_T
/// - u: X coordinate of the centroid
/// - v: Y coordinate of the centroid
/// - s: Size of the bounding box
/// - r: Apect ratio of the bounding box

#[allow(dead_code)]
pub struct ConstantVelocity2DModel<R>
where
    R: RealField,
    DefaultAllocator: Allocator<R, U7, U7>,
    DefaultAllocator: Allocator<R, U7>,
{
    pub transition_model: OMatrix<R, U7, U7>,
    pub transition_model_transpose: OMatrix<R, U7, U7>,
    pub transition_noise_covariance: OMatrix<R, U7, U7>,
}

/// Constant velocity model `x = [ u, v, s, r, v_u, v_v, v_s ]_T`
impl<R> ConstantVelocity2DModel<R>
where
    R: RealField,
{
    #[allow(dead_code)]
    pub fn new() -> Self {
        let one = na::convert(1.0);
        let zero = na::convert(0.0);

        // Note that from_vec uses row major
        #[rustfmt::skip]
        let transition_model = OMatrix::<R, U7, U7>::from_vec(vec![
             one, zero, zero, zero, zero, zero, zero,
            zero,  one, zero, zero, zero, zero, zero,
            zero, zero,  one, zero, zero, zero, zero,
            zero, zero, zero,  one, zero, zero, zero,
             one, zero, zero, zero,  one, zero, zero,
            zero,  one, zero, zero, zero,  one, zero,
            zero, zero,  one, zero, zero, zero,  one]);

        let one_100 = na::convert(0.01);
        let one_10000 = na::convert(0.0001);
        #[rustfmt::skip]
        let transition_noise_covariance = OMatrix::<R,U7,U7>::from_vec(vec![
             one, zero, zero, zero,    zero,        zero,       zero,
            zero,  one, zero, zero,    zero,        zero,       zero,
            zero, zero,  one, zero,    zero,        zero,       zero,
            zero, zero, zero,  one,    zero,        zero,       zero,
            zero, zero, zero, zero, one_100,        zero,       zero,
            zero, zero, zero, zero,    zero,     one_100,       zero,
            zero, zero, zero, zero,    zero,        zero,  one_10000]);

        Self {
            transition_model,
            transition_model_transpose: transition_model.transpose(),
            transition_noise_covariance,
        }
    }
}

impl<R> TransitionModelLinearNoControl<R, U7> for ConstantVelocity2DModel<R>
where
    R: RealField,
    DefaultAllocator: Allocator<R, U7, U7>,
    DefaultAllocator: Allocator<R, U4, U7>,
    DefaultAllocator: Allocator<R, U4, U7>,
    DefaultAllocator: Allocator<R, U4, U4>,
    DefaultAllocator: Allocator<R, U7>,
{
    fn F(&self) -> &OMatrix<R, U7, U7> {
        &self.transition_model
    }
    fn FT(&self) -> &OMatrix<R, U7, U7> {
        &self.transition_model_transpose
    }
    fn Q(&self) -> &OMatrix<R, U7, U7> {
        &self.transition_noise_covariance
    }
}
