# TRACK -> LIGHTGBM PARAMS
NYU_LGBM_PARAMS = {
    "fr_en": dict(
        num_leaves=256,
        learning_rate=0.05,
        min_child_samples=100,     # min_data_in_leaf
        n_estimators=750, # num_boost_round
        cat_smooth=200,
        colsample_bytree=0.7, # feature_fraction
        max_cat_threshold=32,
        objective="binary",
        ),
    "en_es": dict(
        num_leaves=512,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=650,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=32,
        objective="binary",
    ),
    "es_en": dict(
        num_leaves=512,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=600,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=32,
        objective="binary",
    ),
    "all": dict(
        num_leaves=1024,
        learning_rate=0.05,
        min_child_samples=100,
        n_estimators=750,
        cat_smooth=200,
        colsample_bytree=0.7,
        max_cat_threshold=64,
        objective="binary",
    ),
}

CAT_FEATS = {
    # user / session context
    "user_id", "session", "client", "countries", "format",

    #tok identity
    "tok", "lemma", "pos",

    # dep label
    "type",

    # neighbourhood
    "prev_tok", "next_tok", "rt_tok",
    "prev_pos", "next_pos", "rt_pos",
    }

DROP = {
    
    # target, extra
    "label", "is_test", 
    
    # identifiers
    "ex_instance_id", "tok_id", "ex_id", "ex_key",

    # raw meta
    "meta",

    "translation"
    }