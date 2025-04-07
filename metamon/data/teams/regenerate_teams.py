if __name__ == "__main__":
    import shutil
    import os
    import random
    import tqdm
    from metamon.data.team_builder import TeamBuilder

    for gen in range(1, 10):
        for format in ["ubers", "ou", "uu", "ru", "nu", "pu"]:
            print(f"Generating teams for gen{gen}{format}...")
            try:
                tb = TeamBuilder(
                    f"gen{gen}{format}", ps_path="/home/xieleo/pokemon-showdown"
                )
            except FileNotFoundError:
                print(f"gen{gen}{format} not found")
                continue

            dir = os.path.join(f"gen{gen}", format)
            if os.path.exists(dir):
                shutil.rmtree(dir)

            for split in ["train", "test"]:
                os.makedirs(os.path.join(dir, split))

            generated = 0
            while generated <= 1000:
                train_test = "train" if generated <= 900 else "test"
                output_path = os.path.join(dir, train_test, f"team_{generated}")
                team = tb.generate_new_team()
                team = tb.team_to_str(team)

                valid = os.popen(
                    f"echo '{team}' | ~/pokemon-showdown/pokemon-showdown validate-team gen{gen}{format}"
                )
                if valid.close() is None:
                    os.popen(f"echo '{team}' > {output_path}")
                    generated += 1
                    print(f"gen{gen}{format} success")
                else:
                    print(team)
                    print(f"gen{gen}{format} failure")
