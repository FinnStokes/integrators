//! Wrappers for GSL integration routines. See GSL documentation
//! [here](https://www.gnu.org/software/gsl/doc/html/integration.html).
//!
//! This module, and all its re-exports from the top-level `integrators`
//! module, are gated with the `gsl` feature. So, if you don't want to use
//! these wrappers, or don't have GSL installed, just turn off that feature.
//!
//! Note that GSL's integration routines can only support integration over
//! one dimension; if you need to integrate over multiple dimensions, you can
//! nest calls to integrators, or (preferrably) use another library such as
//! Cuba.
//!
//! ```rust
//! use integrators::{Integrator, Real};
//! fn integrate_gaussian(from: f64, to: f64, sigma: f64, mean: f64) -> f64 {
//!     let normalization = (2f64 * ::std::f64::consts::PI).sqrt() * sigma;
//!     integrators::gsl::QAG::new(1000)
//!                       .with_range(from, to)
//!                       .integrate(|x: Real| {
//!                           (-((x - mean) / (2f64 * sigma)).powi(2)).exp()
//!                           / normalization
//!                           }, 1e-6, 1e-18)
//!                       .unwrap()
//!                       .value
//! }
//!
//! let ranges: &'static [(Real, Real)] = &[(-0.3, 0.0), (0.0, 1.5),
//!                                         (1.5, 3.2), (3.2, 5.9)];
//! for &(a, b) in ranges.iter() {
//!     let integrated = integrate_gaussian(a, b, 1.0, 0.0);
//!     println!("range: {}, {}", a, b);
//!     println!("integrated: {}", integrated);
//! }
//! ```

use std::{error, fmt, marker, mem};
use std::convert::{From, Into, TryFrom};
use std::ffi::CStr;
use std::os::raw::{c_void, c_int, c_uint};

use super::bindings;
use super::ffi::LandingPad;
use super::traits::{IntegrandInput, IntegrandOutput};
use super::Real;

#[cfg(test)]
mod test;

mod qng;
pub use self::qng::QNG;

mod qag;
pub use self::qag::{QAG, QAGRule};

mod qags;
pub use self::qags::QAGS;

mod qagp;
pub use self::qagp::QAGP;

mod qagi;
pub use self::qagi::{QAGI, QAGIU, QAGIL};

mod qawf;
pub use self::qawf::QAWF;

unsafe extern "C"
fn gsl_integrand_fn<A, B, F>(x: Real, params: *mut c_void) -> Real
    where A: IntegrandInput,
          B: IntegrandOutput,
          F: FnMut(A) -> B
{
    let fnptr = params as *mut LandingPad<A, B, F>;
    let fun: &mut LandingPad<A, B, F> = &mut *fnptr;

    if A::input_size() != 1 {
        panic!("integrand given to GSL integrator demands >1 input");
    }

    let mut output: [Real; 1] = [0.0];
    match fun.try_call(&[x], &mut output) {
        Ok(_) => output[0],
        Err(_) => 0.0,
    }
}

#[derive(Debug)]
struct GSLFunction<'a> {
    function: bindings::gsl_function,
    lifetime: marker::PhantomData<&'a ()>
}

fn make_gsl_function<'a, A, B, F>(fun: &'a mut LandingPad<A, B, F>, range_low: Real, range_high: Real)
        -> GSLResult<GSLFunction<'a>>
    where A: IntegrandInput,
          B: IntegrandOutput,
          F: FnMut(A) -> B
{
    // Disable the default error handler so we can handle GSL errors in Rust
    // Otherwise - GSL would use default behavior of aborting process on error
    unsafe { bindings::gsl_set_error_handler_off() };

    if A::input_size() != 1 {
        Err(GSLIntegrationError::InvalidInputDim(A::input_size()))
    } else if fun.raw_call(&[(range_low + range_high) / 2f64])
                 .output_size() != 1 {
        let output_size = fun.raw_call(&[(range_low + range_high) / 2f64])
                             .output_size();
        Err(GSLIntegrationError::InvalidOutputDim(output_size))
    } else {
        Ok(GSLFunction {
            function: bindings::gsl_function {
                function: Some(gsl_integrand_fn::<A, B, F>),
                params: unsafe { mem::transmute(fun) }
            },
            lifetime: marker::PhantomData
        })
    }
}

/// Error codes specific to GSL Integration routines. For more information,
/// read the GSL docs
/// [here](https://www.gnu.org/software/gsl/doc/html/integration.html#error-codes)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GSLErrorCode {
    /// Maximum number of iterations has been reached
    MaxIter,
    /// Integration could not converge due to round-off error
    Round,
    /// The integrand contained a non-integrable singularity, or other poor
    /// behavior
    Sing,
    /// The integral is divergent, or too slowly convergent, for the given
    /// algorithm
    Diverge,
    /// Domain error on input arguments
    Domain,
    /// Non-integration-specific GSL error code
    Other(c_int)
}

impl GSLErrorCode {
    /// Yields the raw return code.
    pub fn raw(&self) -> c_int {
        (*self).into()
    }

    pub fn gsl_description(&self) -> Option<&'static str> {
        let descr = unsafe { bindings::gsl_strerror(self.raw()) };
        if descr.is_null() {
            None
        } else {
            Some(unsafe {
                CStr::from_ptr::<'static>(descr)
            }.to_str().expect("gsl_strerror should return valid static string"))
        }
    }
}

impl fmt::Display for GSLErrorCode {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        let n = self.raw();
        if let Some(desc) = self.gsl_description() {
            write!(fmt, "(GSL) error code {}, description: {}", n, desc)
        } else {
            write!(fmt, "(GSL) error code {}", n)
        }
    }
}

impl error::Error for GSLErrorCode {}

impl From<c_int> for GSLErrorCode {
    fn from(n: c_int) -> Self {
        use self::GSLErrorCode::*;
        match n {
            bindings::GSL_EMAXITER => MaxIter,
            bindings::GSL_EROUND => Round,
            bindings::GSL_ESING => Sing,
            bindings::GSL_EDIVERGE => Diverge,
            bindings::GSL_EDOM => Domain,
            _ => Other(n)
        }
    }
}

impl Into<c_int> for GSLErrorCode {
    fn into(self) -> c_int {
        use self::GSLErrorCode::*;
        match self {
            MaxIter => bindings::GSL_EMAXITER,
            Round => bindings::GSL_EROUND,
            Sing => bindings::GSL_ESING,
            Diverge => bindings::GSL_EDIVERGE,
            Domain => bindings::GSL_EDOM,
            Other(n) => n,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GSLIntegrationError {
    InvalidInputDim(usize),
    InvalidOutputDim(usize),
    GSLError(GSLErrorCode),
}

impl fmt::Display for GSLIntegrationError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use self::GSLIntegrationError::*;
        match &self {
            &InvalidInputDim(n) => write!(fmt, "(GSL) Invalid input dim: {}", n),
            &InvalidOutputDim(n) => write!(fmt, "(GSL) Invalid output dim: {}", n),
            &GSLError(err) => write!(fmt, "{}", err)
        }
    }
}

impl error::Error for GSLIntegrationError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self {
            &GSLIntegrationError::GSLError(ref err) => Some(err),
            _ => None,
        }
    }
}

pub type GSLResult<T> = Result<T, GSLIntegrationError>;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct GSLIntegrationResult {
    pub value: Real,
    pub error: Real
}

struct GSLIntegrationWorkspace {
    pub(crate) nintervals: usize,
    wkspc: *mut bindings::gsl_integration_workspace
}

impl fmt::Debug for GSLIntegrationWorkspace {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("GSLIntegrationWorkspace")
           .field("nintervals", &self.nintervals)
           .finish()
    }
}

impl Clone for GSLIntegrationWorkspace {
    fn clone(&self) -> Self {
        GSLIntegrationWorkspace::new(self.nintervals)
    }
}

impl GSLIntegrationWorkspace {
    pub(crate) fn new(n: usize) -> Self {
        GSLIntegrationWorkspace {
            // TODO: Check for null-pointer
            wkspc: unsafe {
                bindings::gsl_integration_workspace_alloc(n)
            },
            nintervals: n
        }
    }
}

impl Drop for GSLIntegrationWorkspace {
    fn drop(&mut self) {
        unsafe {
            bindings::gsl_integration_workspace_free(self.wkspc)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GSLIntegrationQAWOEnum {
    Sine,
    Cosine,
}

impl fmt::Display for GSLIntegrationQAWOEnum {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        use self::GSLIntegrationQAWOEnum::*;
        match self {
            Sine => write!(fmt, "sin"),
            Cosine => write!(fmt, "cos"),
        }
    }
}

impl TryFrom<c_uint> for GSLIntegrationQAWOEnum {
    type Error = &'static str;

    fn try_from(n: c_uint) -> Result<Self, Self::Error> {
        use self::GSLIntegrationQAWOEnum::*;
        match n {
            bindings::gsl_integration_qawo_enum_GSL_INTEG_COSINE => Ok(Cosine),
            bindings::gsl_integration_qawo_enum_GSL_INTEG_SINE => Ok(Sine),
            _ => Err("Invalid enum value for GSLIntegrationQAWO")
        }
    }
}

impl Into<c_uint> for GSLIntegrationQAWOEnum {
    fn into(self) -> c_uint {
        use self::GSLIntegrationQAWOEnum::*;
        match self {
            Cosine => bindings::gsl_integration_qawo_enum_GSL_INTEG_COSINE,
            Sine => bindings::gsl_integration_qawo_enum_GSL_INTEG_SINE,
        }
    }
}
struct GSLIntegrationQAWOTable {
    pub(crate) nintervals: usize,
    pub(crate) interval_length: Real,
    pub(crate) omega: Real,
    pub(crate) sine: GSLIntegrationQAWOEnum,
    table: *mut bindings::gsl_integration_qawo_table
}

impl fmt::Debug for GSLIntegrationQAWOTable {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("GSLIntegrationQAWOTable")
           .field("nintervals", &self.nintervals)
           .field("interval_length", &self.interval_length)
           .field("omega", &self.omega)
           .field("sine", &self.sine)
           .finish()
    }
}

impl Clone for GSLIntegrationQAWOTable {
    fn clone(&self) -> Self {
        GSLIntegrationQAWOTable::new(self.nintervals, self.interval_length, self.omega, self.sine)
    }
}

impl GSLIntegrationQAWOTable {
    pub(crate) fn new(n: usize, interval_length: Real, omega: Real, sine: GSLIntegrationQAWOEnum) -> Self {
        GSLIntegrationQAWOTable {
            // TODO: Check for null-pointer
            table: unsafe {
                bindings::gsl_integration_qawo_table_alloc(
                    omega, interval_length, sine.into(), n,
                )
            },
            nintervals: n,
            interval_length, omega, sine,
        }
    }

    pub(crate) fn with_sin(self, omega: Real) -> Self {
        GSLIntegrationQAWOTable::new(
            self.nintervals, self.interval_length,
            omega, GSLIntegrationQAWOEnum::Sine,
        )
    }

    pub(crate) fn with_cos(self, omega: Real) -> Self {
        GSLIntegrationQAWOTable::new(
            self.nintervals, self.interval_length,
            omega, GSLIntegrationQAWOEnum::Cosine,
        )
    }

    pub(crate) fn with_nintervals(self, n: usize) -> Self {
        GSLIntegrationQAWOTable::new(
            n, self.interval_length,
            self.omega, self.sine,
        )
    }
}

impl Drop for GSLIntegrationQAWOTable {
    fn drop(&mut self) {
        unsafe {
            bindings::gsl_integration_qawo_table_free(self.table)
        }
    }
}
