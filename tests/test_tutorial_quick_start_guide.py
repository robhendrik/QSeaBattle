


import sys
import pandas as pd
sys.path.append("./src")
import Q_Sea_Battle as QSB


# %%
field_sizes = [4,8,16,32]
number_of_games_in_tournament = 10
channel_noise_levels = [0.0, 0.1, 0.3, 0.5]


def test_tutorial_quick_start_guide():
    # %%
    results_list = []
    for field_size in field_sizes:
        for noise_level in channel_noise_levels:
            # Layout: variable field size, 1-bit communication, fixed number of games
            layout = QSB.GameLayout(
                field_size=field_size,
                comms_size=1,
                number_of_games_in_tournament=number_of_games_in_tournament,
                channel_noise=noise_level
            )

            env = QSB.GameEnv(layout)
            players = QSB.Players(layout)
            tournament = QSB.Tournament(env, players, layout)

            log = tournament.tournament()
            mean_reward, std_err = log.outcome()

            # Store results in list
            results_list.append({
                'player_type': 'base',
                'field_size': field_size,
                'noise_level': noise_level,
                'performance': mean_reward,
                '95p error +/-': 1.96 * std_err,
                'reference': 0.5,
                'in_interval': (mean_reward - 1.96 * std_err <= 0.5 <= mean_reward + 1.96 * std_err)
            })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)
    assert results_df is not None

    # %%
    results_list = []
    for field_size in field_sizes:
        for noise_level in channel_noise_levels:
            # Layout: variable field size, 1-bit communication, fixed number of games
            layout = QSB.GameLayout(
                field_size=field_size,
                comms_size=1,
                number_of_games_in_tournament=number_of_games_in_tournament,
                channel_noise=noise_level
            )

            env = QSB.GameEnv(layout)
            players = QSB.SimplePlayers(layout)
            tournament = QSB.Tournament(env, players, layout)

            log = tournament.tournament()
            mean_reward, std_err = log.outcome()

            ref = QSB.expected_win_rate_simple(field_size = field_size, 
                                            comms_size=1,
                                            channel_noise=noise_level)

            # Store results in list
            results_list.append({
                'player_type': 'simple',
                'field_size': field_size,
                'noise_level': noise_level,
                'performance': mean_reward,
                '95p error +/-': 1.96 * std_err,
                'reference': ref,
                'in_interval': (mean_reward - 1.96 * std_err <= ref <= mean_reward + 1.96 * std_err)
            })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)
    assert results_df is not None


    # %%
    results_list = []
    for field_size in field_sizes:
        for noise_level in channel_noise_levels:
            # Layout: variable field size, 1-bit communication, fixed number of games
            layout = QSB.GameLayout(
                field_size=field_size,
                comms_size=1,
                number_of_games_in_tournament=number_of_games_in_tournament,
                channel_noise=noise_level
            )

            env = QSB.GameEnv(layout)
            players = QSB.MajorityPlayers(layout)
            tournament = QSB.Tournament(env, players, layout)

            log = tournament.tournament()
            mean_reward, std_err = log.outcome()

            ref = QSB.expected_win_rate_majority(field_size = field_size, 
                                            comms_size=1,
                                            channel_noise=noise_level)

            # Store results in list
            results_list.append({
                'player_type': 'majority',
                'field_size': field_size,
                'noise_level': noise_level,
                'performance': mean_reward,
                '95p error +/-': 1.96 * std_err,
                'reference': ref,
                'in_interval': (mean_reward - 1.96 * std_err <= ref <= mean_reward + 1.96 * std_err)
            })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)

    assert results_df is not None

    # %%
    results_list = []
    p_high = 1.0
    for field_size in field_sizes:
        for noise_level in channel_noise_levels:
            # Layout: variable field size, 1-bit communication, fixed number of games
            layout = QSB.GameLayout(
                field_size=field_size,
                comms_size=1,
                number_of_games_in_tournament=number_of_games_in_tournament,
                channel_noise=noise_level
            )

            env = QSB.GameEnv(layout)
            players = QSB.PRAssistedPlayers(game_layout = layout, p_high = p_high)
            tournament = QSB.Tournament(env, players, layout)

            log = tournament.tournament()
            mean_reward, std_err = log.outcome()

            ref = QSB.expected_win_rate_assisted(field_size = field_size, 
                                            comms_size=1,
                                            channel_noise=noise_level,
                                            p_high=p_high)

            # Store results in list
            results_list.append({
                'player_type': 'assisted/P_high= '+str(p_high),
                'field_size': field_size,
                'noise_level': noise_level,
                'performance': mean_reward,
                '95p error +/-': 1.96 * std_err,
                'reference': ref,
                'in_interval': (mean_reward - 1.96 * std_err <= ref <= mean_reward + 1.96 * std_err)
            })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)

    assert results_df is not None

    # %%
    results_list = []
    p_high = 0.85
    for field_size in field_sizes:
        for noise_level in channel_noise_levels:
            # Layout: variable field size, 1-bit communication, fixed number of games
            layout = QSB.GameLayout(
                field_size=field_size,
                comms_size=1,
                number_of_games_in_tournament=number_of_games_in_tournament,
                channel_noise=noise_level
            )

            env = QSB.GameEnv(layout)
            players = QSB.PRAssistedPlayers(game_layout = layout, p_high = p_high)
            tournament = QSB.Tournament(env, players, layout)

            log = tournament.tournament()
            mean_reward, std_err = log.outcome()

            ref = QSB.expected_win_rate_assisted(field_size = field_size, 
                                            comms_size=1,
                                            channel_noise=noise_level,
                                            p_high=p_high)

            # Store results in list
            results_list.append({
                'player_type': 'assisted/P_high= '+str(p_high),
                'field_size': field_size,
                'noise_level': noise_level,
                'performance': mean_reward,
                '95p error +/-': 1.96 * std_err,
                'reference': ref,
                'in_interval': (mean_reward - 1.96 * std_err <= ref <= mean_reward + 1.96 * std_err)
            })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)

    assert results_df is not None


    # %%
    results_list = []
    field_size = 64
    noise_level = 0.45
    p_high_values = [float(n/100) for n in range(75,95,1)]
    for p_high in p_high_values:
        # Layout: variable field size, 1-bit communication, fixed number of games
        layout = QSB.GameLayout(
            field_size=field_size,
            comms_size=1,
            number_of_games_in_tournament=number_of_games_in_tournament,
            channel_noise=noise_level
        )

        env = QSB.GameEnv(layout)
        players = QSB.PRAssistedPlayers(game_layout = layout, p_high = p_high)
        tournament = QSB.Tournament(env, players, layout)

        log = tournament.tournament()
        mean_reward, std_err = log.outcome()

        ref = QSB.expected_win_rate_assisted(field_size = field_size, 
                                            comms_size=1,
                                            channel_noise=noise_level,
                                            p_high=p_high)
        
        ic_bound = QSB.limit_from_mutual_information(field_size=field_size,
                                                        comms_size=1,
                                                        channel_noise=noise_level)

        # Store results in list
        results_list.append({
            'player_type': 'assisted/P_high= '+str(p_high),
            'field_size': field_size,
            'noise_level': noise_level,
            'performance': mean_reward,
            'reference': ref,
            'information_constraint': ic_bound,
            'in_interval': (mean_reward - 1.96 * std_err <= ref <= mean_reward + 1.96 * std_err)
        })

    # Create DataFrame from collected results
    results_df = pd.DataFrame(results_list)




