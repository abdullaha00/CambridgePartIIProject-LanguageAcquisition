import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter, defaultdict

rng = np.random.default_rng(seed=42)

ALPHA_ERR = np.array([0.3, 0.1, 0.03, 0.01], dtype=np.float32)
K = len(ALPHA_ERR)


def add_temporal_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    # == Tag and merge into one stream
    
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["is_test"] = 0
    df_test["is_test"] = 1
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    
    #=====================

    user_test_n = df_test.groupby("user_id")["ex_instance_id"].nunique().to_dict()

    #=====================
    
    N = len(df)
    
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

    err_tok = np.zeros((N,K), dtype=np.float32)
    err_root = np.zeros((N,K), dtype=np.float32)

    for uid, g in tqdm(df.groupby("user_id", sort=False), desc="Adding temporal features"):
        # g = g.sort_values("days") assumed

        # EXERCISE encounter

        ex_count = Counter()

        # TOKEN/ROOT state [total enonters, last_seen_time(days), [err1, err2, ...]]

        tok_state = defaultdict(lambda: [0, None, np.zeros(K, dtype=np.float32)]) # tok -> [encounters, last_time_seen, [err1,err2,err3,err4]]
        root_state = defaultdict(lambda: [0, None, np.zeros(K, dtype=np.float32)]) # root -> [encounters, last_time_seen, [err1,err2,err3,err4]]]

        # snapshots of state AFTER each exercise: list of tok/root -> [enc, time, errvec]
        tok_snaps = []
        root_snaps = [] 
        
        n = int(user_test_n.get(uid, 0))
        assert n >= 2

        # IDX of train end
        
        ex_g = g.groupby("ex_instance_id", sort=False)

        train_end_idx = len(ex_g) - n - 1

        for _, ex_df in ex_g:
            
            row_idxs = ex_df.index
            toks = tuple(ex_df["tok"])

            #-----
            idx = len(tok_snaps)
            day = float(ex_df["days"].iloc[0])

            #------ TRAIN-TIME MASK

            # If we are encoding test data, then 
            # last_labeled_idx is the train data end idx

            if idx > train_end_idx:
                last_labeled_idx = train_end_idx
            else:
                n_back = int(rng.integers(1, n)) # 1...n-1
                last_labeled_idx = idx - n_back

            # up-to-date snapshots

            tok_snap_total = tok_snaps[idx-1] if idx > 0 else {}
            root_snap_total = root_snaps[idx-1] if idx > 0 else {}

            # masked snapshots

            if last_labeled_idx >= len(tok_snaps):
                print("last_labeled_idx >= len(tok_snaps)")
                print(last_labeled_idx)
                print(len(tok_snaps))

            tok_state_lab = tok_snaps[last_labeled_idx] if last_labeled_idx >= 0 else {}
            root_state_lab = root_snaps[last_labeled_idx] if last_labeled_idx >= 0 else {}
            
            # --- FEATURE READ ---

            # UNMASKED exercise count
            ex_seen[row_idxs] = ex_count[toks]

            for r, tok, root in zip(row_idxs, ex_df["tok"], ex_df["lemma"]):
                
                # ===== UNMASKED total history

                if tok not in tok_snap_total: # First encounter
                    tok_first[r] = 1.0
                    tok_seen[r] = 0
                    tok_tslast[r] = -99.0
                else:
                    count, time_last = tok_snap_total[tok][:2] 
                    tok_seen[r] = count
                    tok_tslast[r] = day - float(time_last)

                if root not in root_snap_total: # First encounter
                    root_first[r] = 1.0
                    root_seen[r] = 0
                    root_tslast[r] = -99.0
                else:
                    count, time_last = root_snap_total[root][:2] 
                    root_seen[r] = count
                    root_tslast[r] = day - float(time_last)

                # ==== MASKED labeled history

                if tok not in tok_state_lab: # First encounter
                    tok_seen_lab[r] = 0
                    tok_tslast_lab[r] = -99.0
                else:
                    count, time_last, errvec = tok_state_lab[tok] 
                    tok_seen_lab[r] = count
                    tok_tslast_lab[r] = day - float(time_last)
                    err_tok[r, :] = errvec

                if root not in root_state_lab: # First encounter
                    root_seen_lab[r] = 0
                    root_tslast_lab[r] = -99.0
                else:
                    count, time_last, errvec = root_state_lab[root] 
                    root_seen_lab[r] = count
                    root_tslast_lab[r] = day - float(time_last)
                    err_root[r, :] = errvec

                # === derived === 

                tok_seen_unlab[r] = tok_seen[r] - tok_seen_lab[r]
                root_seen_unlab[r] = root_seen[r] - root_seen_lab[r]
            
            # ====== UPDATING INTERNAL COUNTERS (UNMASKED)

            ex_count[toks] += 1

            for utok, tok_df in ex_df.groupby("tok", sort=False):
                count = len(tok_df)

                st = tok_state[utok]

                st[0] += count    # no. encounters
                st[1] = day      # last encounter time
                if idx <= train_end_idx:
                    label_mean = tok_df["label"].mean()
                    st[2] += ALPHA_ERR * (label_mean - st[2])
        
            for root, root_df in ex_df.groupby("lemma", sort=False):
                count = len(root_df)

                st = root_state[root]

                st[0] += count   # no. encounters
                st[1] = day      # last encounter time
                if idx <= train_end_idx:   
                    label_mean = root_df["label"].mean()
                    st[2] += ALPHA_ERR * (label_mean - st[2])

            # UPDATE SNAPSHOTS

            tok_snaps.append({
                k: [v[0], v[1], v[2].copy()]
                for k,v in tok_state.items()
            })

            root_snaps.append({
                k: [v[0], v[1], v[2].copy()]
                for k,v in root_state.items()
            })

    # ======== WRITING TO DF

    df["ex_seen"]         = ex_seen
    df["tok_seen"]        = tok_seen
    df["root_seen"]       = root_seen
    df["tok_seen_lab"]    = tok_seen_lab
    df["root_seen_lab"]   = root_seen_lab
    df["tok_seen_unlab"]  = tok_seen_unlab
    df["root_seen_unlab"] = root_seen_unlab

    df["tok_tslast"]      = tok_tslast
    df["root_tslast"]     = root_tslast
    df["tok_tslast_lab"]  = tok_tslast_lab
    df["root_tslast_lab"] = root_tslast_lab

    df["tok_first"]       = tok_first
    df["root_first"]      = root_first

    for j, a in enumerate(ALPHA_ERR):
        df[f"err_tok_{a}"] = err_tok[:, j]
        df[f"err_root_{a}"] = err_root[:, j]

    # =======

    df_train_out = df[df["is_test"] == 0].reset_index(drop=True)
    df_test_out = df[df["is_test"] == 1].reset_index(drop=True)

    return df_train_out, df_test_out