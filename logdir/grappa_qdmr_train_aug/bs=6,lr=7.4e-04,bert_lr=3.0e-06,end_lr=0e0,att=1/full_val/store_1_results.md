 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **artists** | id | + |   | 
 | 2 |   | name |   |   | 
 | 3 | **sqlite_sequence** | name |   |   | 
 | 4 |   | seq |   |   | 
 | 5 | **albums** | id | + |   | 
 | 6 |   | title |   |   | 
 | 7 |   | artist_id |   | --> 1 | 
 | 8 | **employees** | id | + |   | 
 | 9 |   | last_name |   |   | 
 | 10 |   | first_name |   |   | 
 | 11 |   | title |   |   | 
 | 12 |   | reports_to |   | --> 8 | 
 | 13 |   | birth_date |   |   | 
 | 14 |   | hire_date |   |   | 
 | 15 |   | address |   |   | 
 | 16 |   | city |   |   | 
 | 17 |   | state |   |   | 
 | 18 |   | country |   |   | 
 | 19 |   | postal_code |   |   | 
 | 20 |   | phone |   |   | 
 | 21 |   | fax |   |   | 
 | 22 |   | email |   |   | 
 | 23 | **customers** | id | + |   | 
 | 24 |   | first_name |   |   | 
 | 25 |   | last_name |   |   | 
 | 26 |   | company |   |   | 
 | 27 |   | address |   |   | 
 | 28 |   | city |   |   | 
 | 29 |   | state |   |   | 
 | 30 |   | country |   |   | 
 | 31 |   | postal_code |   |   | 
 | 32 |   | phone |   |   | 
 | 33 |   | fax |   |   | 
 | 34 |   | email |   |   | 
 | 35 |   | support_rep_id |   | --> 8 | 
 | 36 | **genres** | id | + |   | 
 | 37 |   | name |   |   | 
 | 38 | **invoices** | id | + |   | 
 | 39 |   | customer_id |   | --> 23 | 
 | 40 |   | invoice_date |   |   | 
 | 41 |   | billing_address |   |   | 
 | 42 |   | billing_city |   |   | 
 | 43 |   | billing_state |   |   | 
 | 44 |   | billing_country |   |   | 
 | 45 |   | billing_postal_code |   |   | 
 | 46 |   | total |   |   | 
 | 47 | **media_types** | id | + |   | 
 | 48 |   | name |   |   | 
 | 49 | **tracks** | id | + |   | 
 | 50 |   | name |   |   | 
 | 51 |   | album_id |   | --> 5 | 
 | 52 |   | media_type_id |   | --> 47 | 
 | 53 |   | genre_id |   | --> 36 | 
 | 54 |   | composer |   |   | 
 | 55 |   | milliseconds |   |   | 
 | 56 |   | bytes |   |   | 
 | 57 |   | unit_price |   |   | 
 | 58 | **invoice_lines** | id | + |   | 
 | 59 |   | invoice_id |   | --> 38 | 
 | 60 |   | track_id |   | --> 49 | 
 | 61 |   | unit_price |   |   | 
 | 62 |   | quantity |   |   | 
 | 63 | **playlists** | id | + |   | 
 | 64 |   | name |   |   | 
 | 65 | **playlist_tracks** | playlist_id | + | --> 63 | 
 | 66 |   | track_id |   | --> 49 | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_551 | List every album's title. | SELECT title FROM albums; | 1. SELECT[tbl:​albums] <br>2. PROJECT[col:​albums:​title, #1] <br> | 1. SELECT[tbl:​albums] <br>2. PROJECT[col:​albums:​title, #1] <br> | + | easy | 
  | SPIDER_train_552 | What are the titles of all the albums? | SELECT title FROM albums; | 1. SELECT[tbl:​albums] <br>2. PROJECT[col:​albums:​title, #1] <br> | 1. SELECT[tbl:​albums] <br>2. PROJECT[col:​albums:​title, #1] <br> | + | easy | 
  | SPIDER_train_563 | List the number of invoices from the US, grouped by state. | SELECT billing_state ,  COUNT(*) FROM invoices WHERE billing_country  =  "USA" GROUP BY billing_state; | 1. SELECT[tbl:​invoices] <br>2. FILTER[#1, comparative:​=:​USA:​col:​invoices:​billing_country] <br>3. PROJECT[col:​invoices:​billing_state, #2] <br>4. GROUP[count, #1, #3] <br>5. UNION[#3, #4] <br> | 1. SELECT[tbl:​invoices] <br>2. COMPARATIVE[#1, #1, comparative:​=:​USA:​col:​invoices:​billing_country] <br>3. PROJECT[col:​invoices:​billing_state, #2] <br>4. GROUP[count, #1, #3] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_train_565 | List the state in the US with the most invoices. | SELECT billing_state ,  COUNT(*) FROM invoices WHERE billing_country  =  "USA" GROUP BY billing_state ORDER BY COUNT(*) DESC LIMIT 1; | 1. SELECT[col:​invoices:​billing_state] <br>2. FILTER[#1, comparative:​=:​USA:​col:​invoices:​billing_country] <br>3. PROJECT[tbl:​invoices, #2] <br>4. GROUP[count, #3, #1] <br>5. SUPERLATIVE[comparative:​max:​None, #1, #4] <br>6. PROJECT[tbl:​invoices, #5] <br>7. GROUP[count, #6, #5] <br>8. UNION[#5, #7] <br> | 1. SELECT[col:​invoices:​billing_state] <br>2. COMPARATIVE[#1, #1, comparative:​=:​USA:​col:​invoices:​billing_country] <br>3. PROJECT[tbl:​invoices, #2] <br>4. GROUP[count, #3, #1] <br>5. SUPERLATIVE[comparative:​max:​None, #1, #4] <br>6. PROJECT[tbl:​invoices, #5] <br>7. GROUP[count, #6, #5] <br>8. UNION[#5, #7] <br> | + | extra | 
  | SPIDER_train_569 | List Aerosmith's albums. | SELECT T1.title FROM albums AS T1 JOIN artists AS T2 ON  T1.artist_id = T2.id WHERE T2.name = "Aerosmith"; | 1. SELECT[val:​artists:​name:​Aerosmith] <br>2. PROJECT[col:​albums:​title, #1] <br> | 1. SELECT[val:​artists:​name:​Aerosmith] <br>2. PROJECT[col:​albums:​title, #1] <br> | + | medium | 
  | SPIDER_train_570 | What are the titles of all the Aerosmith albums? | SELECT T1.title FROM albums AS T1 JOIN artists AS T2 ON  T1.artist_id = T2.id WHERE T2.name = "Aerosmith"; | 1. SELECT[val:​artists:​name:​Aerosmith] <br>2. PROJECT[tbl:​albums, #1] <br>3. PROJECT[col:​albums:​title, #2] <br> | 1. SELECT[val:​artists:​name:​Aerosmith] <br>2. PROJECT[tbl:​albums, #1] <br>3. PROJECT[col:​albums:​title, #2] <br> | + | medium | 
  | SPIDER_train_571 | How many albums does Billy Cobham has? | SELECT count(*) FROM albums AS T1 JOIN artists AS T2 ON  T1.artist_id = T2.id WHERE T2.name = "Billy Cobham"; | 1. SELECT[val:​artists:​name:​Billy Cobham] <br>2. PROJECT[tbl:​albums, #1] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[val:​tracks:​composer:​Billy Cobham] <br>2. PROJECT[tbl:​albums, #1] <br>3. AGGREGATE[count, #2] <br> | - | medium | 
  | SPIDER_train_572 | How many albums has Billy Cobam released? | SELECT count(*) FROM albums AS T1 JOIN artists AS T2 ON  T1.artist_id = T2.id WHERE T2.name = "Billy Cobham"; | 1. SELECT[tbl:​albums] <br>2. FILTER[#1, comparative:​=:​Billy Cobham:​col:​artists:​name] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​albums] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Billy Cobham:​col:​artists:​name] <br>3. AGGREGATE[count, #2] <br> | + | medium | 
  | SPIDER_train_577 | How many customers live in Prague city? | SELECT count(*) FROM customers WHERE city = "Prague"; | 1. SELECT[tbl:​customers] <br>2. FILTER[#1, comparative:​=:​Prague:​col:​customers:​city] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Prague:​col:​customers:​city] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_578 | How many customers live in the city of Prague? | SELECT count(*) FROM customers WHERE city = "Prague"; | 1. SELECT[tbl:​customers] <br>2. FILTER[#1, comparative:​=:​Prague:​col:​customers:​city] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Prague:​col:​customers:​city] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_579 | How many customers in state of CA? | SELECT count(*) FROM customers WHERE state = "CA"; | 1. SELECT[tbl:​customers] <br>2. FILTER[#1, comparative:​=:​CA:​col:​customers:​state] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​CA:​col:​customers:​state] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_580 | How many customers are from California? | SELECT count(*) FROM customers WHERE state = "CA"; | 1. SELECT[tbl:​customers] <br>2. FILTER[#1, comparative:​=:​CA:​col:​customers:​state] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​customers] <br>2. COMPARATIVE[#1, #1, comparative:​=:​CA:​col:​customers:​state] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_583 | List the name of albums that are released by aritist whose name has 'Led' | SELECT T2.title FROM artists AS T1 JOIN albums AS T2 ON T1.id  =  T2.artist_id WHERE T1.name LIKE '%Led%' | 1. SELECT[tbl:​artists] <br>2. PROJECT[col:​artists:​name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​%Led%:​col:​artists:​name] <br>4. PROJECT[tbl:​albums, #3] <br>5. PROJECT[col:​albums:​title, #4] <br> | 1. SELECT[tbl:​artists] <br>2. PROJECT[col:​artists:​name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​Led:​col:​artists:​name] <br>4. PROJECT[tbl:​albums, #3] <br>5. PROJECT[col:​albums:​title, #4] <br> | + | hard | 
  | SPIDER_train_584 | What is the title of the album that was released by the artist whose name has the phrase 'Led'? | SELECT T2.title FROM artists AS T1 JOIN albums AS T2 ON T1.id  =  T2.artist_id WHERE T1.name LIKE '%Led%' | 1. SELECT[tbl:​artists] <br>2. PROJECT[col:​artists:​name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​%Led%:​col:​artists:​name] <br>4. PROJECT[tbl:​albums, #3] <br>5. PROJECT[col:​albums:​title, #4] <br> | 1. SELECT[tbl:​artists] <br>2. PROJECT[col:​artists:​name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​like:​Led:​col:​artists:​name] <br>4. PROJECT[tbl:​albums, #3] <br>5. PROJECT[col:​albums:​title, #4] <br> | + | hard | 
  | SPIDER_train_595 | How many employees are living in Canada? | SELECT count(*) FROM employees WHERE country = "Canada"; | 1. SELECT[tbl:​employees] <br>2. FILTER[#1, comparative:​=:​Canada:​col:​employees:​country] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​employees] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Canada:​col:​employees:​country] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_596 | How many employees live in Canada? | SELECT count(*) FROM employees WHERE country = "Canada"; | 1. SELECT[tbl:​employees] <br>2. FILTER[#1, comparative:​=:​Canada:​col:​employees:​country] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​employees] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Canada:​col:​employees:​country] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_611 | List all media types. | SELECT name FROM media_types; | 1. SELECT[col:​media_types:​name] <br> | 1. SELECT[col:​media_types:​name] <br> | + | easy | 
  | SPIDER_train_612 | What are the names of all the media types? | SELECT name FROM media_types; | 1. SELECT[tbl:​media_types] <br>2. PROJECT[col:​media_types:​name, #1] <br> | 1. SELECT[tbl:​media_types] <br>2. PROJECT[col:​media_types:​name, #1] <br> | + | easy | 
  | SPIDER_train_613 | List all different genre types. | SELECT DISTINCT name FROM genres; | 1. SELECT[tbl:​genres] <br>2. PROJECT[col:​genres:​name, #1] <br>3.*(distinct)* PROJECT[different #REF, #2] <br> | 1. SELECT[tbl:​genres] <br>2. PROJECT[col:​genres:​name, #1] <br>3.*(distinct)* PROJECT[None, #2] <br> | + | easy | 
  | SPIDER_train_614 | What are the different names of the genres? | SELECT DISTINCT name FROM genres; | 1. SELECT[tbl:​genres] <br>2.*(distinct)* PROJECT[col:​genres:​name, #1] <br> | 1. SELECT[tbl:​genres] <br>2.*(distinct)* PROJECT[col:​genres:​name, #1] <br> | + | easy | 
  | SPIDER_train_615 | List the name of all playlist. | SELECT name FROM playlists; | 1. SELECT[tbl:​playlists] <br>2. PROJECT[col:​playlists:​name, #1] <br> | 1. SELECT[tbl:​playlists] <br>2. PROJECT[col:​playlists:​name, #1] <br> | + | easy | 
  | SPIDER_train_616 | What are the names of all the playlists? | SELECT name FROM playlists; | 1. SELECT[tbl:​playlists] <br>2. PROJECT[col:​playlists:​name, #1] <br> | 1. SELECT[tbl:​playlists] <br>2. PROJECT[col:​playlists:​name, #1] <br> | + | easy | 
  | SPIDER_train_617 | Who is the composer of track Fast As a Shark? | SELECT composer FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[tbl:​tracks] <br>2. FILTER[#1, comparative:​=:​Fast As a Shark:​col:​tracks:​name] <br>3. PROJECT[col:​tracks:​composer, #2] <br> | 1. SELECT[tbl:​tracks] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Fast As a Shark:​col:​tracks:​name] <br>3. PROJECT[col:​tracks:​composer, #2] <br> | + | easy | 
  | SPIDER_train_618 | What is the composer who created the track "Fast As a Shark"? | SELECT composer FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​composer, #1] <br> | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​composer, #1] <br> | + | easy | 
  | SPIDER_train_619 | How long does track Fast As a Shark has? | SELECT milliseconds FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​milliseconds, #1] <br> | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​milliseconds, #1] <br> | + | easy | 
  | SPIDER_train_620 | How many milliseconds long is Fast As a Shark? | SELECT milliseconds FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​milliseconds, #1] <br>3. FILTER[#2, col:​tracks:​milliseconds] <br> | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​milliseconds, #1] <br>3. COMPARATIVE[#2, #2, col:​tracks:​milliseconds] <br> | + | easy | 
  | SPIDER_train_621 | What is the name of tracks whose genre is Rock? | SELECT T2.name FROM genres AS T1 JOIN tracks AS T2 ON T1.id = T2.genre_id WHERE T1.name = "Rock"; | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | - | medium | 
  | SPIDER_train_622 | What is the name of all tracks in the Rock genre? | SELECT T2.name FROM genres AS T1 JOIN tracks AS T2 ON T1.id = T2.genre_id WHERE T1.name = "Rock"; | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | - | medium | 
  | SPIDER_train_629 | List the name of tracks belongs to genre Rock and whose media type is MPEG audio file. | SELECT T2.name FROM genres AS T1 JOIN tracks AS T2 ON T1.id = T2.genre_id JOIN media_types AS T3 ON T3.id = T2.media_type_id WHERE T1.name = "Rock" AND T3.name = "MPEG audio file"; | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[tbl:​media_types, #1] <br>5. COMPARATIVE[#1, #4, comparative:​=:​MPEG audio file:​col:​media_types:​name] <br>6. INTERSECTION[#1, #3, #5] <br>7. PROJECT[col:​tracks:​name, #6] <br> | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. PROJECT[tbl:​media_types, #1] <br>5. COMPARATIVE[#1, #4, comparative:​=:​MPEG audio file:​col:​media_types:​name] <br>6. INTERSECTION[#1, #3, #5] <br>7. PROJECT[col:​tracks:​name, #6] <br> | - | hard | 
  | SPIDER_train_633 | List the name of tracks belongs to genre Rock or genre Jazz. | SELECT T2.name FROM genres AS T1 JOIN tracks AS T2 ON T1.id = T2.genre_id WHERE T1.name = "Rock" OR T1.name = "Jazz" | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Jazz:​col:​genres:​name] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | 1. SELECT[tbl:​tracks] <br>2. PROJECT[tbl:​genres, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Rock:​col:​genres:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Jazz:​col:​genres:​name] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | - | hard | 
  | SPIDER_train_634 | What are the names of the tracks that are Rock or Jazz songs? | SELECT T2.name FROM genres AS T1 JOIN tracks AS T2 ON T1.id = T2.genre_id WHERE T1.name = "Rock" OR T1.name = "Jazz" | 1. SELECT[tbl:​tracks] <br>2. FILTER[#1, comparative:​=:​Rock:​col:​genres:​name] <br>3. FILTER[#1, comparative:​=:​Jazz:​col:​genres:​name] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​tracks:​name, #4] <br> | 1. SELECT[tbl:​tracks] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Rock:​col:​genres:​name] <br>3. COMPARATIVE[#1, #1, comparative:​=:​Jazz:​col:​genres:​name] <br>4. UNION[#2, #3] <br>5. PROJECT[col:​tracks:​name, #4] <br> | - | hard | 
  | SPIDER_train_635 | List the name of all tracks in the playlists of Movies. | SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id = T2.track_id JOIN playlists AS T3 ON T3.id = T2.playlist_id WHERE T3.name = "Movies"; | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​Movies:​col:​playlists:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​Movies:​col:​playlists:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | + | hard | 
  | SPIDER_train_636 | What are the names of all tracks that are on playlists titled Movies? | SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id = T2.track_id JOIN playlists AS T3 ON T3.id = T2.playlist_id WHERE T3.name = "Movies"; | 1. SELECT[tbl:​playlist_tracks] <br>2. FILTER[#1, tbl:​playlist_tracks] <br>3. FILTER[#2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | 1. SELECT[tbl:​playlist_tracks] <br>2. COMPARATIVE[#1, #1, tbl:​playlist_tracks] <br>3. COMPARATIVE[#2, #2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. PROJECT[col:​tracks:​name, #3] <br> | + | hard | 
  | SPIDER_train_641 | How much is the track Fast As a Shark? | SELECT unit_price FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[tbl:​tracks] <br>2. FILTER[#1, comparative:​=:​Fast As a Shark:​col:​tracks:​name] <br>3. PROJECT[col:​tracks:​unit_price, #2] <br> | 1. SELECT[tbl:​tracks] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Fast As a Shark:​col:​tracks:​name] <br>3. PROJECT[col:​tracks:​unit_price, #2] <br> | + | easy | 
  | SPIDER_train_642 | What is the unit price of the tune "Fast As a Shark"? | SELECT unit_price FROM tracks WHERE name = "Fast As a Shark"; | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​unit_price, #1] <br> | 1. SELECT[val:​tracks:​name:​Fast As a Shark] <br>2. PROJECT[col:​tracks:​unit_price, #1] <br> | + | easy | 
  | SPIDER_train_644 | What are the names of all tracks that are on the Movies playlist but not in the music playlist? | SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Movies' EXCEPT SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Music' | 1. SELECT[tbl:​playlist_tracks] <br>2. FILTER[#1, comparative:​=:​Movies:​col:​playlists:​name] <br>3. FILTER[#1, comparative:​=:​Music:​col:​playlists:​name] <br>4. INTERSECTION[#1, #2, #3] <br>5. DISCARD[#2, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | 1. SELECT[tbl:​playlist_tracks] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Movies:​col:​playlists:​name] <br>3. COMPARATIVE[#1, #1, comparative:​=:​Music:​col:​playlists:​name] <br>4. INTERSECTION[#1, #2, #3] <br>5. DISCARD[#2, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | + | extra | 
  | SPIDER_train_645 | Find the name of tracks which are in both Movies and music playlists. | SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Movies' INTERSECT SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Music' | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Music:​col:​playlists:​name] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Music:​col:​playlists:​name] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | + | extra | 
  | SPIDER_train_646 | What are the names of all the tracks that are in both the Movies and music playlists? | SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Movies' INTERSECT SELECT T1.name FROM tracks AS T1 JOIN playlist_tracks AS T2 ON T1.id  =  T2.track_id JOIN playlists AS T3 ON T2.playlist_id  =  T3.id WHERE T3.name  =  'Music' | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Music:​col:​playlists:​name] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | 1. SELECT[tbl:​playlist_tracks] <br>2. PROJECT[tbl:​playlist_tracks, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Movies:​col:​playlists:​name] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Music:​col:​playlists:​name] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​tracks:​name, #5] <br> | + | extra | 
 ***
 Exec acc: **0.8421**
