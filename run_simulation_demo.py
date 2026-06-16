from stochastic_simulation import SimulationParams, run_replications, simulate_one


def main() -> None:
    params = SimulationParams(
        T=100.0,
        warmup=20.0,
        c=52,
        lam=623.3,
        mu_plus=13.5,
        mu_minus=13.5,
        thetaA=4.0,
        thetaS=3.0,
        thetaL=3.0,
        deltaB=1 / 130,
        deltaS=3.0,
        deltaL=1 / 9,
        gamma=1 / 260,
        q0=0,
        b0=0,
        rs0=0,
        rl0=0,
        seed=123,
    )

    result = simulate_one(params)
    print(result)

    frame = run_replications(params, 5)
    print(
        frame[
            [
                "replication",
                "seed",
                "total_arrival_rate_per_model_day",
                "average_wait_including_abandonments_minutes",
                "average_speed_to_answer_minutes",
                "abandonment_fraction",
                "procedural_denial_rate_per_model_day",
            ]
        ]
    )


if __name__ == "__main__":
    main()
