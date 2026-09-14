# Production prediction output specification

The production system predicts three separate objects from strictly point-in-time-safe pre-match information:

1. **1X2:** Home / Draw / Away probabilities.
2. **MOM (Man of the Match): exactly 4 players:** rank the four highest-probability eligible players. The candidate universe must be restricted to players who are eligible to participate according to pre-match information. Post-match ratings, in-match statistics, final MOM labels, or any data published only after kickoff are prohibited as features.
3. **Scoreline: exactly 3 scorelines:** rank the three highest-probability scorelines from the score distribution (Poisson/Dixon-Coles candidates may supply the distribution).

Top-k output is presentation/output policy; the underlying probabilities must remain available for evaluation. Candidate selection and production adoption remain subject to chronological OOS validation, calibration, PIT checks, and the existing adoption gate.
