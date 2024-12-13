from enum import Enum

class Turn(Enum):
    NIGHT = 1
    MORNING = 2
    AFTERNOON = 3
    FREE = 0
    NUM_DAILY_TURNS = 4
    NUM_WEEKLY_TURNS = 21