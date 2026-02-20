from bisect import bisect_right
import logging
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

rng = np.random.default_rng(seed=42)

ALPHA_ERR = np.array([0.3, 0.1, 0.03, 0.01], dtype=np.float32)
K = len(ALPHA_ERR)
INIT_TSLAST = -99.0

def ex_key_global(df: pd.DataFrame) -> np.ndarray:
    return df["tok_id"].str.slice(0, 10).to_numpy(dtype=object)


def history_lookup(history_idxs: dict, history_state: dict, key: str, targ_idx: int):
    idxs = history_idxs.get(key)
    if idxs is None:
        return None

    pos = bisect_right(idxs, targ_idx) - 1
    if pos == -1:  # no history idx <= targ_idx
        return None

    return history_state[key][pos]


def add_temporal_features_stream(df_all: pd.DataFrame) -> pd.DataFrame:

    df = df_all
    N = len(df)

    # == use numpy indexing

    tok_arr = df["tok"].to_numpy(dtype=object)
    root_arr = df["lemma"].to_numpy(dtype=object)
    day_arr = df["days"].to_numpy(dtype=np.float32, copy=False)
    label_arr = df["label"].to_numpy(dtype=np.float32, copy=False)
    is_test_arr = df["is_test"].to_numpy(dtype=np.int8, copy=False)
    ex_key_arr = ex_key_global(df)

    # ===== OUTPUTS ======

    # UNMASKED total encounters
    ex_seen = np.zeros(N, dtype=np.int32)
    tok_seen = np.zeros(N, dtype=np.int32)
    root_seen = np.zeros(N, dtype=np.int32)

    # MASKED labeled encounters
    tok_seen_lab = np.zeros(N, dtype=np.int32)
    root_seen_lab = np.zeros(N, dtype=np.int32)

    # DERIVED unlabelled encounters
    tok_seen_unlab = np.zeros(N, dtype=np.int32)
    root_seen_unlab = np.zeros(N, dtype=np.int32)

    # UNMASKED time since last encounter
    tok_tslast = np.full(N, -99.0, dtype=np.float32)
    root_tslast = np.full(N, -99.0, dtype=np.float32)

    # MASKED time since last label
    tok_tslast_lab = np.full(N, -99.0, dtype=np.float32)
    root_tslast_lab = np.full(N, -99.0, dtype=np.float32)

    # FIRST encounter flag
    tok_first = np.zeros(N, dtype=np.float32)
    root_first = np.zeros(N, dtype=np.float32)

    # MASKED error average
    err_tok = np.zeros((N, K), dtype=np.float32)
    err_root = np.zeros((N, K), dtype=np.float32)

    # uid -> [idx1, idx2, ...]
    user_groups = df.groupby("user_id", sort=False).indices

    for uid, user_rows in tqdm(user_groups.items(), desc="Adding temporal features"):
        U = user_rows.size

        if U == 0:
            logger.warning("User with no data: %s", uid)

        # unique ex keys for user
        u_ex = ex_key_arr[user_rows]
        ex_start = np.where(np.r_[True, u_ex[1:] != u_ex[:-1]])[0]
        ex_ends = np.r_[ex_start[1:], U]

        n_ex = ex_start.size

        n_test = is_test_arr[user_rows[ex_start]].sum()
        if n_test == 0:
            logger.warning("User with no test data: %s", uid)

        train_end_idx = n_ex - n_test - 1

        # ===== PASS 1: FULL TEMPORAL HISTORIES and EX_SEEN with NO MASKING
        ex_count = Counter()

        tok_count = Counter()
        tok_last_day: dict[str, float] = {}
        tok_erravg: dict[str, np.ndarray] = {}

        root_count = Counter()
        root_last_day: dict[str, float] = {}
        root_erravg: dict[str, np.ndarray] = {}

        tok_hist_idx: dict[
            str, list[int]
        ] = {}  # tok -> [idx1, idx2, ...] in encounter order (exercise idx)
        tok_hist_state: dict[
            str, list[tuple[int, float, np.ndarray]]
        ] = {}  # tok -> [(count, last_day, errvec), ...] in encounter order
        root_hist_idx: dict[str, list[int]] = {}
        root_hist_state: dict[str, list[tuple[int, float, np.ndarray]]] = {}

        # Iterate over each exercise
        for idx, (start, end) in enumerate(zip(ex_start, ex_ends)):
            is_train = idx <= train_end_idx

            row_idxs = user_rows[start:end]  # now in global df idx space

            ex_toks = tok_arr[row_idxs]
            ex_roots = root_arr[row_idxs]
            ex_labels = label_arr[row_idxs]

            toks_tuple = tuple(ex_toks)
            day = day_arr[row_idxs[0]]  # all rows in ex have same day

            ex_seen[row_idxs] = ex_count[toks_tuple]
            ex_count[toks_tuple] += 1

            tok_agg: dict[str, list[int]] = {}  # tok -> [count, label_sum]
            root_agg: dict[str, list[int]] = {}  # tok -> [count, label_sum]

            # == AGGREGATE COUNTS AND LABELS
            for tok, root, label in zip(ex_toks, ex_roots, ex_labels):
                # -- counts, labels
                if tok in tok_agg:
                    tok_agg[tok][0] += 1
                    if is_train:
                        tok_agg[tok][1] += label
                else:
                    tok_agg[tok] = [1, label if is_train else 0]

                if root in root_agg:
                    root_agg[root][0] += 1
                    if is_train:
                        root_agg[root][1] += label
                else:
                    root_agg[root] = [1, label if is_train else 0]

            # UPDATE TOKEN/ROOT STATE FOLLOWING AGGREGATION
            for tok, (count, label_sum) in tok_agg.items():
                tok_count[tok] += count
                tok_last_day[tok] = day

                # -- ERRVEC
                errvec = tok_erravg.get(tok)
                if errvec is None:
                    errvec = np.zeros(K, dtype=np.float32)
                    tok_erravg[tok] = errvec
                if is_train:
                    label_mean = label_sum / count
                    errvec += ALPHA_ERR * (label_mean - errvec)

                if tok not in tok_hist_idx:
                    tok_hist_idx[tok] = [idx]
                    tok_hist_state[tok] = [(tok_count[tok], day, errvec.copy())]
                else:
                    tok_hist_idx[tok].append(idx)
                    tok_hist_state[tok].append((tok_count[tok], day, errvec.copy()))

            for root, (count, label_sum) in root_agg.items():
                root_count[root] += count
                root_last_day[root] = day

                # -- ERRVEC
                errvec = root_erravg.get(root)
                if errvec is None:
                    errvec = np.zeros(K, dtype=np.float32)
                    root_erravg[root] = errvec
                if is_train:
                    label_mean = label_sum / count
                    errvec += ALPHA_ERR * (label_mean - errvec)

                if root not in root_hist_idx:
                    root_hist_idx[root] = [idx]
                    root_hist_state[root] = [(root_count[root], day, errvec.copy())]
                else:
                    root_hist_idx[root].append(idx)
                    root_hist_state[root].append((root_count[root], day, errvec.copy()))

        # =========== PASS 2: MASKING
        # At this point, we have full history available
        for idx, (start, end) in enumerate(zip(ex_start, ex_ends)):
            row_idxs = user_rows[start:end]  # now in global df idx space

            ex_toks = tok_arr[row_idxs]
            ex_roots = root_arr[row_idxs]
            ex_labels = label_arr[row_idxs]

            day = day_arr[row_idxs[0]]  # all rows in ex have same day

            # if test, we see lables up to train_end_idx
            # if train, we mask with n_back in [1, n_test-1]
            is_test = idx > train_end_idx
            if is_test:
                last_labeled_idx = train_end_idx
            else:
                if n_test == 0:
                    last_labeled_idx = idx - 1
                else:
                    n_back = int(rng.integers(0, n_test))  # 1...n-1
                    last_labeled_idx = idx - n_back - 1

            for row, tok, root in zip(row_idxs, ex_toks, ex_roots):
                # ===== UPDATE UNMASKED
                tok_prev_state = history_lookup(
                    tok_hist_idx, tok_hist_state, tok, idx - 1
                )

                if tok_prev_state is None:
                    tok_first[row] = 1.0
                else:
                    count_prev, day_prev, errvec_prev = tok_prev_state
                    tok_seen[row] = count_prev
                    tok_tslast[row] = day - day_prev

                root_prev_state = history_lookup(
                    root_hist_idx, root_hist_state, root, idx - 1
                )
                if root_prev_state is None:
                    root_first[row] = 1.0
                else:
                    count_prev, day_prev, errvec_prev = root_prev_state
                    root_seen[row] = count_prev
                    root_tslast[row] = day - day_prev

                # ===== UPDATE MASKED
                if last_labeled_idx >= 0:
                    tok_prev_state_lab = history_lookup(
                        tok_hist_idx, tok_hist_state, tok, last_labeled_idx
                    )
                    if tok_prev_state_lab is not None:
                        count_prev_lab, day_prev_lab, errvec_prev_lab = (
                            tok_prev_state_lab
                        )

                        tok_seen_lab[row] = count_prev_lab
                        tok_tslast_lab[row] = day - day_prev_lab
                        err_tok[row, :] = errvec_prev_lab

                    root_prev_state_lab = history_lookup(
                        root_hist_idx, root_hist_state, root, last_labeled_idx
                    )
                    if root_prev_state_lab is not None:
                        count_prev_lab, day_prev_lab, errvec_prev_lab = (
                            root_prev_state_lab
                        )

                        root_seen_lab[row] = count_prev_lab
                        root_tslast_lab[row] = day - day_prev_lab
                        err_root[row, :] = errvec_prev_lab

                # UNLABELED DERIVED
                tok_seen_unlab[row] = tok_seen[row] - tok_seen_lab[row]
                root_seen_unlab[row] = root_seen[row] - root_seen_lab[row]

    # ======== WRITING TO DF

    df["ex_seen"] = ex_seen
    df["tok_seen"] = tok_seen
    df["root_seen"] = root_seen
    df["tok_seen_lab"] = tok_seen_lab
    df["root_seen_lab"] = root_seen_lab
    df["tok_seen_unlab"] = tok_seen_unlab
    df["root_seen_unlab"] = root_seen_unlab

    df["tok_tslast"] = tok_tslast
    df["root_tslast"] = root_tslast
    df["tok_tslast_lab"] = tok_tslast_lab
    df["root_tslast_lab"] = root_tslast_lab

    df["tok_first"] = tok_first
    df["root_first"] = root_first

    for j, a in enumerate(ALPHA_ERR):
        df[f"err_tok_{a:.3f}"] = err_tok[:, j]
        df[f"err_root_{a:.3f}"] = err_root[:, j]

    # =======

    return df


def add_temporal_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_all = pd.concat(
        [df_train.assign(is_test=0), df_test.assign(is_test=1)], ignore_index=True
    )
    df_all = add_temporal_features_stream(df_all)
    df_train_out = df_all[df_all["is_test"] == 0].reset_index(drop=True)
    df_test_out = df_all[df_all["is_test"] == 1].reset_index(drop=True)
    return df_train_out, df_test_out
