use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    BruteForce(runner::brute_force::MyArgs),
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::BruteForce(my_args) => {
            runner::brute_force::start(my_args, env!("CARGO_PKG_VERSION")).await?
        }
    }
    Ok(())
}
