pub mod decision;
pub mod diffusion;
pub mod message;
pub mod user;

pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
