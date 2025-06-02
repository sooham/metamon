# Replay Parser

The replay parser converts from the spectator POV of raw Pokémon Showdown replays to the first-peron POV of RL agents. The process is summarized by the following (simplified) example:

<p align="center">
  <img src="../../../../media/replay_reconstruction_example.png">
</p>

For a real example, view [this replay](https://replay.Pokémonshowdown.com/gen4nu-776588848). The parser begins with the raw replay:

<p align="center">
  <img src="../../../../media/raw_replay_example.png" width="400">
</p>

And infers missing information to produce training data like:

<p align="center">
  <img src="../../../../media/reconstructed_replay_example.png" width="400">
</p>

More information and discussion in Appendix D of the paper.

### Roadmap
- Improved team inference logic (beyond all-time averages of Showdown usage stats) ✅ 
- Support arbitrary observation spaces / reward functions ✅
- Constrain EV guesses based on damage outcomes and move orders
- Support for Gen1-4 random battles
