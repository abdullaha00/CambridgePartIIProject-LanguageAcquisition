from typing import Dict, Set


FEATURE_GROUPS: Dict[str, Set[str]] = {
    "user_id": {"user_id"},
    "user_other": {

        # user.py
        "burst_mean", 
        "burst_median", 
        "burst_count", 
        "tod_entropy",

        # metadata in dataset
        "client",
        "countries",
    },

    "word_id": {
        
    }

    
    }