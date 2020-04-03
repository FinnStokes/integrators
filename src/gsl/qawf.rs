use ::bindings;
use ::{IntegrationResult, Integrator, Real};
use ::ffi::LandingPad;
use ::traits::{IntegrandInput, IntegrandOutput};

use super::{make_gsl_function, GSLIntegrationError, GSLIntegrationWorkspace,
            GSLIntegrationQAWOTable, GSLIntegrationQAWOEnum};

/// Quadrature Adaptive Weighted integration for Fourier integrals. Computes a
/// Fourier integral over a semi-infinite interval. The integral is computed
/// using the QAWO algorithm over a set of subintervals chosen to cover an odd
/// number of periods so that the contributions from the intervals alternate in
/// sign and are monotonically decreasing when the integration function is
/// positive and monotonically decreasing. The sum of this sequence of
/// contributions is accelerated using the epsilon-algorithm.
#[derive(Debug, Clone)]
pub struct QAWF {
    lower_bound: Real,
    wkspc: GSLIntegrationWorkspace,
    cycle_wkspc: GSLIntegrationWorkspace,
    table: GSLIntegrationQAWOTable,
}

impl QAWF {
    /// Creates a new QAWF with enough memory for `nintervals` subintervals.
    /// This will create a QAWF to integrate the range (0, inf) with the weight
    /// function sin(x). To change the integration range or the weight function,
    /// see `with_bound`, `with_sin` and `with_cos`.
    pub fn new(nintervals: usize) -> Self {
        QAWF {
            lower_bound: 0.0,
            wkspc: GSLIntegrationWorkspace::new(nintervals),
            cycle_wkspc: GSLIntegrationWorkspace::new(nintervals),
            table: GSLIntegrationQAWOTable::new(nintervals, 1.0, 1.0, GSLIntegrationQAWOEnum::Sine),
        }
    }

    /// Discards the old workspaces and allocates new ones with enough memory
    /// for `nintervals` subintervals.
    pub fn with_nintervals(self, nintervals: usize) -> Self {
        QAWF {
            wkspc: GSLIntegrationWorkspace::new(nintervals),
            cycle_wkspc: GSLIntegrationWorkspace::new(nintervals),
            table: self.table.with_nintervals(nintervals),
            ..self
        }
    }

    /// Updates the integration range to ( `lower_bound` , inf)
    pub fn with_bound(self, lower_bound: Real) -> Self {
        QAWF { lower_bound, ..self }
    }

    /// Updates the weight function to sin( `omega` x)
    pub fn with_sin(self, omega: Real) -> Self {
        QAWF { table: self.table.with_sin(omega), ..self }
    }

    /// Updates the weight function to cos( `omega` x)
    pub fn with_cos(self, omega: Real) -> Self {
        QAWF { table: self.table.with_cos(omega), ..self }
    }
}

impl Integrator for QAWF {
    type Success = IntegrationResult;
    type Failure = GSLIntegrationError;
    fn integrate<A, B, F: FnMut(A) -> B>(&mut self, fun: F, _epsrel: Real, epsabs: Real) -> Result<Self::Success, Self::Failure>
        where A: IntegrandInput,
              B: IntegrandOutput
    {
        let mut value: Real = 0.0;
        let mut error: Real = 0.0;

        let mut lp = LandingPad::new(fun);
        let retcode = unsafe {
            let mut gslfn = make_gsl_function(&mut lp, self.lower_bound, self.lower_bound + 1.0)?;
            bindings::gsl_integration_qawf(&mut gslfn.function,
                                           self.lower_bound,
                                           epsabs,
                                           self.wkspc.nintervals,
                                           self.wkspc.wkspc,
                                           self.cycle_wkspc.wkspc,
                                           self.table.table,
                                           &mut value,
                                           &mut error)
        };
        lp.maybe_resume_unwind();

        if retcode != bindings::GSL_SUCCESS {
            Err(GSLIntegrationError::GSLError(retcode.into()))
        } else {
            Ok(IntegrationResult {
                value, error
            })
        }
    }
}
