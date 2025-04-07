import os
import re
import numpy as np
from difflib import get_close_matches
from metamon.data.team_builder import TeamBuilder


gen = "gen3"
mods = "ou"
threshold = 1600
team_folder = f"/data3/grigsby/shared_pokemon_project/parsed_replays_v3/{gen}{mods}/"

# find all the npz files under the folder recursively
npz_files = []
for root, dirs, files in os.walk(team_folder):
    for file in files:
        if file.endswith(".npz"):
            npz_files.append(os.path.join(root, file))

# all the file are named tier-time_score_state.npz, parse the score
scores = {}
for npz_file in npz_files:
    file_name = os.path.basename(npz_file)
    score, win = file_name.split("_")[1], file_name.split("_")[2]
    if win == "LOSS.npz":
        continue
    if score not in scores:
        scores[score] = []
    scores[score].append(npz_file)

# find all files with score
file = []
for score in scores.keys():
    if score == "Unrated":
        continue
    if int(score) > threshold:
        file += scores[score]

print(f"Found {len(file)} files with score > {threshold}")

movesets_data_file = (
    f"/home/xieleo/work/metamon/metamon/data/movesets_data/{gen}/inclusive.json"
)
import json

with open(movesets_data_file) as f:
    movesets_data = json.load(f)

official_move_names = set()
official_item_names = set()
official_ability_names = set()
for pm in movesets_data.keys():
    official_move_names.update(movesets_data[pm]["moves"].keys())
    official_item_names.update(movesets_data[pm]["items"].keys())
    official_ability_names.update(movesets_data[pm]["abilities"].keys())

official_pm_names = set(movesets_data.keys())
official_pm_names = list(official_pm_names)
official_pm_names.sort()
official_move_names = list(official_move_names)
official_move_names.sort()
official_item_names = list(official_item_names)
official_item_names.sort()
official_ability_names = list(official_ability_names)
official_ability_names.sort()

print(f"Found {len(official_pm_names)} official pokemon names")
print(f"Found {len(official_move_names)} official move names")
print(f"Found {len(official_item_names)} official item names")
print(f"Found {len(official_ability_names)} official ability names")

# lower case all the moves, and remove special characters
official_move_names_l = {
    m.lower().replace(" ", "").replace("-", "").replace(".", ""): m
    for m in official_move_names
}
official_pm_names_l = {
    m.lower().replace(" ", "").replace("-", "").replace(".", ""): m
    for m in official_pm_names
}
official_item_names_l = {
    m.lower().replace(" ", "").replace("-", "").replace(".", ""): m
    for m in official_item_names
}
official_ability_names_l = {
    m.lower().replace(" ", "").replace("-", "").replace(".", ""): m
    for m in official_ability_names
}

teambuilder = TeamBuilder(
    f"{gen}{mods}", ps_path="/home/xieleo/pokemon-showdown", verbose=False
)


def get_best_match(move, list):
    matches = get_close_matches(move, list, n=1, cutoff=0.6)
    if not matches:
        raise ValueError(f"No match found for {move}")
    return matches[0]


def parse_pokemon_string(poke_string):
    movesets = {}
    match = re.search(
        r"<player> (\w+) (\w+) (\w+) .*?<move> (\w+) .*?<move> (\w+) .*?<move> (\w+) .*?<move> (\w+)",
        poke_string,
    )
    if not match:
        return "Invalid string: active Pokémon or moves missing"

    active_pokemon = official_pm_names_l[
        get_best_match(match.group(1), official_pm_names_l.keys())
    ]
    active_item = official_item_names_l[
        get_best_match(match.group(2), official_item_names_l.keys())
    ]
    active_ability = official_ability_names_l[
        get_best_match(match.group(3), official_ability_names_l.keys())
    ]
    active_moves = [
        official_move_names_l[
            get_best_match(match.group(i), official_move_names_l.keys())
        ]
        for i in range(4, 8)
    ]
    active_moves.sort()

    active_pm = teambuilder.generate_partial_moveset(
        pokemon=active_pokemon,
        item=active_item,
        ability=active_ability,
        selected_moves=active_moves,
    )

    switch_matches = re.findall(
        r"<switch> (\w+) (\w+) (\w+) .*?<moveset> (\w+) (\w+) (\w+) (\w+)", poke_string
    )

    for switch in switch_matches:
        poke_name, poke_item, poke_ability, move1, move2, move3, move4 = switch
        pm_name = official_pm_names_l[
            get_best_match(poke_name, official_pm_names_l.keys())
        ]
        pm_item = official_item_names_l[
            get_best_match(poke_item, official_item_names_l.keys())
        ]
        pm_ability = official_ability_names_l[
            get_best_match(poke_ability, official_ability_names_l.keys())
        ]
        move1 = official_move_names_l[
            get_best_match(move1, official_move_names_l.keys())
        ]
        move2 = official_move_names_l[
            get_best_match(move2, official_move_names_l.keys())
        ]
        move3 = official_move_names_l[
            get_best_match(move3, official_move_names_l.keys())
        ]
        move4 = official_move_names_l[
            get_best_match(move4, official_move_names_l.keys())
        ]
        moves = [move1, move2, move3, move4]
        moves.sort()

        pm = teambuilder.generate_partial_moveset(
            pokemon=pm_name, item=pm_item, ability=pm_ability, selected_moves=moves
        )
        movesets[pm_name] = pm

    team = [active_pm] + list(movesets.values())
    return teambuilder.team_to_str(team), active_pokemon


unique_teams = set()
unique_leads = set()
lead_sets = {}
movesets = set()
for f in file:
    data = np.load(f)
    try:
        team, active_pm = parse_pokemon_string(str(data["obs_text"][0]))
    except ValueError as e:
        print(f"Error in {f}: {e}")
        continue
    unique_teams.add(team)
    unique_leads.add(active_pm)
    if active_pm not in lead_sets:
        lead_sets[active_pm] = set()

print("Here's an example of a parsed team:")
print(parse_pokemon_string(str(data["obs_text"][0]))[0])


print("From: ", len(file), " to: ", len(unique_teams))

print("Unique team leads:")
for lead in unique_leads:
    print(lead)

print("Saving unique teams to file")
os.makedirs(
    f"/home/xieleo/work/metamon/metamon/data/teams/{gen}/{mods}/replays", exist_ok=True
)
for i, team in enumerate(unique_teams):
    with open(
        f"/home/xieleo/work/metamon/metamon/data/teams/{gen}/{mods}/replays/team_{i}",
        "w",
    ) as f:
        f.write(team)
