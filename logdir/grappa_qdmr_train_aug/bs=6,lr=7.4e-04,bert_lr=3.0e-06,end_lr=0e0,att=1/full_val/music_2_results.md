 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Songs** | SongId | + |   | 
 | 2 |   | Title |   |   | 
 | 3 | **Albums** | AId | + |   | 
 | 4 |   | Title |   |   | 
 | 5 |   | Year |   |   | 
 | 6 |   | Label |   |   | 
 | 7 |   | Type |   |   | 
 | 8 | **Band** | Id | + |   | 
 | 9 |   | Firstname |   |   | 
 | 10 |   | Lastname |   |   | 
 | 11 | **Instruments** | SongId | + | --> 1 | 
 | 12 |   | BandmateId |   | --> 8 | 
 | 13 |   | Instrument |   |   | 
 | 14 | **Performance** | SongId | + | --> 1 | 
 | 15 |   | Bandmate |   | --> 8 | 
 | 16 |   | StagePosition |   |   | 
 | 17 | **Tracklists** | AlbumId | + | --> 3 | 
 | 18 |   | Position |   |   | 
 | 19 |   | SongId |   | --> 1 | 
 | 20 | **Vocals** | SongId | + | --> 1 | 
 | 21 |   | Bandmate |   | --> 8 | 
 | 22 |   | Type |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_5172 | How many bands are there? | SELECT count(*) FROM Band | 1. SELECT[tbl:​Band] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Band] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_5173 | Find the number of bands. | SELECT count(*) FROM Band | 1. SELECT[tbl:​Band] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Band] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_5174 | What are all the labels? | SELECT DISTINCT label FROM Albums | 1. SELECT[col:​Albums:​Label] <br> | 1. SELECT[col:​Albums:​Label] <br> | + | easy | 
  | SPIDER_train_5175 | What are the different album labels listed? | SELECT DISTINCT label FROM Albums | 1. SELECT[tbl:​Albums] <br>2. PROJECT[col:​Albums:​Label, #1] <br>3.*(distinct)* PROJECT[different #REF, #2] <br> | 1. SELECT[tbl:​Albums] <br>2. PROJECT[col:​Albums:​Label, #1] <br>3.*(distinct)* PROJECT[None, #2] <br> | + | easy | 
  | SPIDER_train_5180 | How many songs are there? | SELECT count(*) FROM Songs | 1. SELECT[tbl:​Songs] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Songs] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_5181 | Count the number of songs. | SELECT count(*) FROM Songs | 1. SELECT[tbl:​Songs] <br>2. AGGREGATE[count, #1] <br> | 1. SELECT[tbl:​Songs] <br>2. AGGREGATE[count, #1] <br> | + | easy | 
  | SPIDER_train_5192 | How many unique labels are there for albums? | SELECT count(DISTINCT label) FROM albums | 1. SELECT[tbl:​Albums] <br>2. PROJECT[col:​Albums:​Label, #1] <br>3. PROJECT[col:​Albums:​Label, #2] <br>4. AGGREGATE[count, #3] <br> | 1. SELECT[tbl:​Albums] <br>2. PROJECT[col:​Albums:​Label, #1] <br>3. PROJECT[col:​Albums:​Label, #2] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_train_5194 | What is the label that has the most albums? | SELECT label FROM albums GROUP BY label ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Albums:​Label] <br>2. PROJECT[tbl:​Albums, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​Albums:​Label] <br>2. PROJECT[tbl:​Albums, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_5195 | What is the label with the most albums? | SELECT label FROM albums GROUP BY label ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Albums:​Label] <br>2. PROJECT[tbl:​Albums, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​Albums:​Label] <br>2. PROJECT[tbl:​Albums, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_5200 | Find all the songs whose name contains the word "the". | SELECT title FROM songs WHERE title LIKE '% the %' | 1. SELECT[col:​Songs:​Title] <br>2. PROJECT[col:​Songs:​Title, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​% the %:​col:​Songs:​Title] <br> | 1. SELECT[col:​Songs:​Title] <br>2. PROJECT[col:​Songs:​Title, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​ the :​col:​Songs:​Title] <br> | + | medium | 
  | SPIDER_train_5201 | What are the names of the songs whose title has the word "the"? | SELECT title FROM songs WHERE title LIKE '% the %' | 1. SELECT[tbl:​Songs] <br>2. PROJECT[col:​Songs:​Title, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​% the %:​col:​Songs:​Title] <br>4. PROJECT[col:​Songs:​Title, #3] <br> | 1. SELECT[tbl:​Songs] <br>2. PROJECT[col:​Songs:​Title, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​ the :​col:​Songs:​Title] <br>4. PROJECT[col:​Songs:​Title, #3] <br> | + | medium | 
  | SPIDER_train_5202 | What are all the instruments used? | SELECT DISTINCT instrument FROM Instruments | 1. SELECT[col:​Instruments:​Instrument] <br> | 1. SELECT[col:​Instruments:​Instrument] <br> | + | easy | 
  | SPIDER_train_5203 | What are the different instruments listed in the database? | SELECT DISTINCT instrument FROM Instruments | 1. SELECT[col:​Instruments:​Instrument] <br>2. FILTER[#1, listed in the database] <br>3.*(distinct)* PROJECT[different #REF, #2] <br> | 1. SELECT[col:​Instruments:​Instrument] <br>2. COMPARATIVE[#1, #1, None] <br>3.*(distinct)* PROJECT[None, #2] <br> | + | easy | 
  | SPIDER_train_5206 | What is the most used instrument? | SELECT instrument FROM instruments GROUP BY instrument ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Instruments:​Instrument] <br>2. PROJECT[used #REF, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​Instruments:​Instrument] <br>2. PROJECT[None, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | hard | 
  | SPIDER_train_5207 | What instrument is used the most? | SELECT instrument FROM instruments GROUP BY instrument ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Instruments:​Instrument] <br>2. FILTER[#1, comparative:​max:​None] <br> | 1. SELECT[col:​Instruments:​Instrument] <br>2. SUPERLATIVE[comparative:​max:​None, #1, #1] <br> | + | hard | 
  | SPIDER_train_5218 | Which song has the most vocals? | SELECT title FROM vocals AS T1 JOIN songs AS T2 ON T1.songid  =  T2.songid GROUP BY T1.songid ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Songs:​Title] <br>2. SELECT[tbl:​Vocals] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | 1. SELECT[col:​Songs:​Title] <br>2. SELECT[tbl:​Vocals] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | extra | 
  | SPIDER_train_5219 | What is the song with the most vocals? | SELECT title FROM vocals AS T1 JOIN songs AS T2 ON T1.songid  =  T2.songid GROUP BY T1.songid ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Songs:​Title] <br>2. PROJECT[tbl:​Vocals, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​max:​None] <br> | 1. SELECT[col:​Songs:​Title] <br>2. PROJECT[tbl:​Vocals, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br> | + | extra | 
  | SPIDER_train_5220 | Which vocal type is the most frequently appearring type? | SELECT TYPE FROM vocals GROUP BY TYPE ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[col:​Vocals:​Type] <br>2. FILTER[#1, comparative:​max:​None] <br> | 1. SELECT[col:​Vocals:​Type] <br>2. SUPERLATIVE[comparative:​max:​None, #1, #1] <br> | + | hard | 
  | SPIDER_train_5234 | Find all the vocal types. | SELECT DISTINCT TYPE FROM vocals | 1. SELECT[tbl:​Vocals] <br>2. PROJECT[col:​Vocals:​Type, #1] <br> | 1. SELECT[tbl:​Vocals] <br>2. PROJECT[col:​Vocals:​Type, #1] <br> | + | easy | 
  | SPIDER_train_5235 | What are the different types of vocals? | SELECT DISTINCT TYPE FROM vocals | 1. SELECT[tbl:​Vocals] <br>2.*(distinct)* PROJECT[col:​Vocals:​Type, #1] <br> | 1. SELECT[tbl:​Vocals] <br>2.*(distinct)* PROJECT[col:​Vocals:​Type, #1] <br> | + | easy | 
  | SPIDER_train_5258 | Find the first name of the band mate that has performed in most songs. | SELECT t2.firstname FROM Performance AS t1 JOIN Band AS t2 ON t1.bandmate  =  t2.id JOIN Songs AS T3 ON T3.SongId  =  T1.SongId GROUP BY firstname ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​Band] <br>2. PROJECT[tbl:​Songs, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Band:​Firstname, #4] <br> | 1. SELECT[tbl:​Band] <br>2. PROJECT[tbl:​Songs, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Band:​Firstname, #4] <br> | + | extra | 
  | SPIDER_train_5259 | What is the first name of the band mate who perfomed in the most songs? | SELECT t2.firstname FROM Performance AS t1 JOIN Band AS t2 ON t1.bandmate  =  t2.id JOIN Songs AS T3 ON T3.SongId  =  T1.SongId GROUP BY firstname ORDER BY count(*) DESC LIMIT 1 | 1. SELECT[tbl:​Band] <br>2. PROJECT[tbl:​Songs, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Band:​Firstname, #4] <br> | 1. SELECT[tbl:​Band] <br>2. PROJECT[tbl:​Songs, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​Band:​Firstname, #4] <br> | + | extra | 
 ***
 Exec acc: **1.0000**
