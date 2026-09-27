# Opta-like free/public source frontier

The project now tracks a research-only registry of free/public football
data sources and feature engines that can complement the current ESPN,
SofaScore, Football-Data and Global Football Data Lake stack.

The registry intentionally separates:

- source/data availability from historical PIT validity;
- data providers from feature/normalization engines;
- high-value research candidates from production eligibility.

No source is automatically promoted. Historical use requires a verified
publication/availability timestamp or a conservative fail-closed policy.

Initial high-value channels include:

1. event-level data (StatsBomb/Wyscout)
2. 360 spatial context (StatsBomb 360)
3. broadcast tracking and dynamic events (SkillCorner)
4. tracking/event synchronization (Metrica)
5. video-derived tracking benchmarks (SoccerNet)
6. Opta-derived public tables (FBref)
7. independent shot/xG data (Understat)
8. roster/injury/transfer context (Transfermarkt)
9. action-value engines (socceraction: xT/VAEP)
10. multi-provider normalization (kloppy)
11. pitch control / EPV / velocity feature implementations (LaurieOnTracking)

See `src/research/opta_like_source_registry.py`.
