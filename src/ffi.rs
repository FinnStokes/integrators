//! This module is meant to help users implement their own Foreign Function
//! Interfaces. Currently, the only facility it provides for that is
//! `LandingPad`, which helps to safely catch panics in a Rust integrand
//! to prevent unwinding through foreign code and causing undefined behavior.
//! Once foreign code has finished calling Rust code, the `LandingPad` can
//! be inspected to see if it has caught any panics, and resume the panic
//! safely through Rust code.

use std::any::Any;
use std::marker::PhantomData;
use std::panic;
use ::traits::{IntegrandInput, IntegrandOutput};
use ::Real;

pub struct LandingPad<A, B, F: FnMut(A) -> B> {
    err: Option<Box<dyn Any + Send + 'static>>,
    fun: F,
    a: PhantomData<A>,
    b: PhantomData<B>,
}

impl<A: IntegrandInput, B: IntegrandOutput, F: FnMut(A) -> B> LandingPad<A, B, F> {
    pub fn new(fun: F) -> Self {
        LandingPad {
            err: None, fun: fun,
            a: PhantomData, b: PhantomData,
        }
    }

    /// Attempts to apply the integrand to the given `args` and write to
    /// `output`. If any error happens, including a panic in IntegrandInput
    /// or IntegrandOutput, the function resturns `Err(_)`, if the integrand
    /// runs normally, returns `Ok(())`.
    ///
    /// If the integrand panics once, and `try_call()` is called again later,
    /// the `LandingPad` will return a reference to the result of the earlier
    /// panic. In other words, the integrand will only be allowed to panic
    /// once.
    pub fn try_call(&mut self, args: &[Real], output: &mut [Real]) -> Result<(), &(dyn Any + Send + 'static)> {
        if self.err.is_some() {
            Err(self.err.as_ref().expect("just said it is some"))
        } else {
            // FIXME: Is there a better way to do this? This took some hassle to
            // figure out
            let res = {
                let mut borself = panic::AssertUnwindSafe(&mut *self);
                let mut output_buf = panic::AssertUnwindSafe(&mut *output);
                panic::catch_unwind(move || {
                    let lp: &mut LandingPad<A, B, F> = &mut borself;
                    (lp.fun)(A::from_args(args)).into_args(&mut *output_buf)
                })
            };
            match res {
                Ok(()) => Ok(()),
                Err(err) => {
                    self.err = Some(err);
                    Err(self.err.as_ref().expect("just set to Some(..)"))
                }
            }
        }
    }

    pub fn raw_call(&mut self, args: &[Real]) -> B {
        (self.fun)(A::from_args(args))
    }

    pub fn maybe_resume_unwind(self) {
        if self.err.is_some() {
            self.resume_unwind()
        }
    }

    pub fn finish(self) -> Option<Box<dyn Any + Send + 'static>> {
        self.err
    }

    fn resume_unwind(self) -> ! {
        panic::resume_unwind(self.err.expect("trying to resume unwind"))
    }
}
