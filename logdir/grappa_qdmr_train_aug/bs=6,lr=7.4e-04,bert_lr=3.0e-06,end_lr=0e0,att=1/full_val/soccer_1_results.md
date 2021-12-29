 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **sqlite_sequence** | name |   |   | 
 | 2 |   | seq |   |   | 
 | 3 | **Player_Attributes** | id | + |   | 
 | 4 |   | player_fifa_api_id |   | --> 48 | 
 | 5 |   | player_api_id |   | --> 46 | 
 | 6 |   | date |   |   | 
 | 7 |   | overall_rating |   |   | 
 | 8 |   | potential |   |   | 
 | 9 |   | preferred_foot |   |   | 
 | 10 |   | attacking_work_rate |   |   | 
 | 11 |   | defensive_work_rate |   |   | 
 | 12 |   | crossing |   |   | 
 | 13 |   | finishing |   |   | 
 | 14 |   | heading_accuracy |   |   | 
 | 15 |   | short_passing |   |   | 
 | 16 |   | volleys |   |   | 
 | 17 |   | dribbling |   |   | 
 | 18 |   | curve |   |   | 
 | 19 |   | free_kick_accuracy |   |   | 
 | 20 |   | long_passing |   |   | 
 | 21 |   | ball_control |   |   | 
 | 22 |   | acceleration |   |   | 
 | 23 |   | sprint_speed |   |   | 
 | 24 |   | agility |   |   | 
 | 25 |   | reactions |   |   | 
 | 26 |   | balance |   |   | 
 | 27 |   | shot_power |   |   | 
 | 28 |   | jumping |   |   | 
 | 29 |   | stamina |   |   | 
 | 30 |   | strength |   |   | 
 | 31 |   | long_shots |   |   | 
 | 32 |   | aggression |   |   | 
 | 33 |   | interceptions |   |   | 
 | 34 |   | positioning |   |   | 
 | 35 |   | vision |   |   | 
 | 36 |   | penalties |   |   | 
 | 37 |   | marking |   |   | 
 | 38 |   | standing_tackle |   |   | 
 | 39 |   | sliding_tackle |   |   | 
 | 40 |   | gk_diving |   |   | 
 | 41 |   | gk_handling |   |   | 
 | 42 |   | gk_kicking |   |   | 
 | 43 |   | gk_positioning |   |   | 
 | 44 |   | gk_reflexes |   |   | 
 | 45 | **Player** | id | + |   | 
 | 46 |   | player_api_id |   |   | 
 | 47 |   | player_name |   |   | 
 | 48 |   | player_fifa_api_id |   |   | 
 | 49 |   | birthday |   |   | 
 | 50 |   | height |   |   | 
 | 51 |   | weight |   |   | 
 | 52 | **League** | id | + |   | 
 | 53 |   | country_id |   | --> 55 | 
 | 54 |   | name |   |   | 
 | 55 | **Country** | id | + |   | 
 | 56 |   | name |   |   | 
 | 57 | **Team** | id | + |   | 
 | 58 |   | team_api_id |   |   | 
 | 59 |   | team_fifa_api_id |   |   | 
 | 60 |   | team_long_name |   |   | 
 | 61 |   | team_short_name |   |   | 
 | 62 | **Team_Attributes** | id | + |   | 
 | 63 |   | team_fifa_api_id |   | --> 59 | 
 | 64 |   | team_api_id |   | --> 58 | 
 | 65 |   | date |   |   | 
 | 66 |   | buildUpPlaySpeed |   |   | 
 | 67 |   | buildUpPlaySpeedClass |   |   | 
 | 68 |   | buildUpPlayDribbling |   |   | 
 | 69 |   | buildUpPlayDribblingClass |   |   | 
 | 70 |   | buildUpPlayPassing |   |   | 
 | 71 |   | buildUpPlayPassingClass |   |   | 
 | 72 |   | buildUpPlayPositioningClass |   |   | 
 | 73 |   | chanceCreationPassing |   |   | 
 | 74 |   | chanceCreationPassingClass |   |   | 
 | 75 |   | chanceCreationCrossing |   |   | 
 | 76 |   | chanceCreationCrossingClass |   |   | 
 | 77 |   | chanceCreationShooting |   |   | 
 | 78 |   | chanceCreationShootingClass |   |   | 
 | 79 |   | chanceCreationPositioningClass |   |   | 
 | 80 |   | defencePressure |   |   | 
 | 81 |   | defencePressureClass |   |   | 
 | 82 |   | defenceAggression |   |   | 
 | 83 |   | defenceAggressionClass |   |   | 
 | 84 |   | defenceTeamWidth |   |   | 
 | 85 |   | defenceTeamWidthClass |   |   | 
 | 86 |   | defenceDefenderLineClass |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_1293 | List all country and league names. | SELECT T1.name ,  T2.name FROM Country AS T1 JOIN League AS T2 ON T1.id  =  T2.country_id | 1. SELECT[tbl:​Country] <br>2. PROJECT[col:​Country:​name, #1] <br>3. PROJECT[tbl:​League, #1] <br>4. PROJECT[col:​League:​name, #3] <br>5. UNION[#2, #4] <br> | 1. SELECT[tbl:​Country] <br>2. PROJECT[col:​Country:​name, #1] <br>3. PROJECT[tbl:​League, #1] <br>4. PROJECT[col:​League:​name, #3] <br>5. UNION[#2, #4] <br> | + | medium | 
  | SPIDER_train_1294 | How many leagues are there in England? | SELECT count(*) FROM Country AS T1 JOIN League AS T2 ON T1.id  =  T2.country_id WHERE T1.name  =  "England" | 1. SELECT[tbl:​League] <br>2. FILTER[#1, comparative:​=:​England:​col:​Country:​name] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​League] <br>2. COMPARATIVE[#1, #1, comparative:​=:​England:​col:​Country:​name] <br>3. AGGREGATE[count, #2] <br> | + | medium | 
  | SPIDER_train_1295 | What is the average weight of all players? | SELECT avg(weight) FROM Player | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player:​weight, #1] <br>3. AGGREGATE[avg, #2] <br> | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player:​weight, #1] <br>3. AGGREGATE[avg, #2] <br> | + | easy | 
  | SPIDER_train_1296 | What is the maximum and minimum height of all players? | SELECT max(weight) ,  min(weight) FROM Player | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player:​weight, #1] <br>3. AGGREGATE[min, #2] <br>4. AGGREGATE[max, #2] <br>5. UNION[#4, #3] <br> | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player:​weight, #1] <br>3. AGGREGATE[min, #2] <br>4. AGGREGATE[max, #2] <br>5. UNION[#4, #3] <br> | + | medium | 
  | SPIDER_train_1297 | List all player names who have an overall rating higher than the average. | SELECT DISTINCT T1.player_name FROM Player AS T1 JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T2.overall_rating  >  ( SELECT avg(overall_rating) FROM Player_Attributes ) | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player_Attributes:​overall_rating, #1] <br>3. AGGREGATE[avg, #2] <br>4. COMPARATIVE[#1, #2, comparative:​>:​#3] <br>5. PROJECT[col:​Player:​player_name, #4] <br> | 1. SELECT[tbl:​Player] <br>2. PROJECT[col:​Player_Attributes:​overall_rating, #1] <br>3. AGGREGATE[avg, #2] <br>4. COMPARATIVE[#1, #2, comparative:​>:​right:​col:​Player_Attributes:​preferred_foot] <br>5. PROJECT[col:​Player:​player_name, #4] <br> | - | extra | 
  | SPIDER_train_1299 | List the names of all players who have a crossing score higher than 90 and prefer their right foot. | SELECT DISTINCT T1.player_name FROM Player AS T1 JOIN Player_Attributes AS T2 ON T1.player_api_id = T2.player_api_id WHERE T2.crossing  >  90 AND T2.preferred_foot  =  "right" | 1. SELECT[tbl:​Player_Attributes] <br>2. PROJECT[col:​Player_Attributes:​crossing, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​90:​col:​Player_Attributes:​crossing] <br>4. PROJECT[col:​Player_Attributes:​preferred_foot, #1] <br>5. COMPARATIVE[#1, #4, comparative:​=:​right:​col:​Player_Attributes:​preferred_foot] <br>6. INTERSECTION[#1, #3, #5] <br>7. PROJECT[col:​Player:​player_name, #6] <br> | 1. SELECT[tbl:​Player_Attributes] <br>2. PROJECT[col:​Player_Attributes:​crossing, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​90:​col:​Player_Attributes:​crossing] <br>4. PROJECT[col:​Player_Attributes:​preferred_foot, #1] <br>5. COMPARATIVE[#1, #4, comparative:​=:​right:​col:​Player_Attributes:​preferred_foot] <br>6. INTERSECTION[#1, #3, #5] <br>7. PROJECT[col:​Player:​player_name, #6] <br> | + | medium | 
  | SPIDER_train_1303 | List all of the player ids with a height of at least 180cm and an overall rating higher than 85. | SELECT player_api_id FROM Player WHERE height  >=  180 INTERSECT SELECT player_api_id FROM Player_Attributes WHERE overall_rating  >  85 | 1. SELECT[col:​Player:​player_api_id] <br>2. PROJECT[col:​Player:​height, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>=:​180:​col:​Player:​height] <br>4. PROJECT[col:​Player_Attributes:​overall_rating, #1] <br>5. COMPARATIVE[#1, #4, comparative:​>:​85:​col:​Player_Attributes:​overall_rating] <br>6. INTERSECTION[#1, #3, #5] <br> | 1. SELECT[col:​Player:​player_api_id] <br>2. PROJECT[col:​Player:​height, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>=:​180:​col:​Player:​height] <br>4. PROJECT[col:​Player_Attributes:​overall_rating, #1] <br>5. COMPARATIVE[#1, #4, comparative:​>:​85:​col:​Player_Attributes:​overall_rating] <br>6. INTERSECTION[#1, #3, #5] <br> | + | hard | 
  | SPIDER_train_1304 | List all of the ids for left-footed players with a height between 180cm and 190cm. | SELECT player_api_id FROM Player WHERE height  >=  180 AND height  <=  190 INTERSECT SELECT player_api_id FROM Player_Attributes WHERE preferred_foot  =  "left" | 1. SELECT[col:​Player:​player_api_id] <br>2. FILTER[#1, comparative:​=:​left:​col:​Player_Attributes:​preferred_foot] <br>3. PROJECT[col:​Player:​height, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​180:​col:​Player:​height] <br>5. COMPARATIVE[#1, #3, comparative:​<=:​190:​col:​Player:​height] <br>6. INTERSECTION[#1, #4, #5] <br> | 1. SELECT[col:​Player:​player_api_id] <br>2. COMPARATIVE[#1, #1, comparative:​=:​left:​col:​Player_Attributes:​preferred_foot] <br>3. PROJECT[col:​Player:​height, #2] <br>4. COMPARATIVE[#1, #3, comparative:​>=:​180:​col:​Player:​height] <br>5. COMPARATIVE[#1, #3, comparative:​<=:​190:​col:​Player:​height] <br>6. INTERSECTION[#1, #4, #5] <br> | + | extra | 
 ***
 Exec acc: **0.8750**
