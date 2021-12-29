 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **works_on** | Essn | + | --> 7 | 
 | 2 |   | Pno |   | --> 19 | 
 | 3 |   | Hours |   |   | 
 | 4 | **employee** | Fname |   |   | 
 | 5 |   | Minit |   |   | 
 | 6 |   | Lname |   |   | 
 | 7 |   | Ssn | + |   | 
 | 8 |   | Bdate |   |   | 
 | 9 |   | Address |   |   | 
 | 10 |   | Sex |   |   | 
 | 11 |   | Salary |   |   | 
 | 12 |   | Super_ssn |   | --> 7 | 
 | 13 |   | Dno |   |   | 
 | 14 | **department** | Dname |   |   | 
 | 15 |   | Dnumber | + |   | 
 | 16 |   | Mgr_ssn |   | --> 7 | 
 | 17 |   | Mgr_start_date |   |   | 
 | 18 | **project** | Pname |   |   | 
 | 19 |   | Pnumber | + |   | 
 | 20 |   | Plocation |   |   | 
 | 21 |   | Dnum |   |   | 
 | 22 | **dependent** | Essn | + | --> 7 | 
 | 23 |   | Dependent_name |   |   | 
 | 24 |   | Sex |   |   | 
 | 25 |   | Bdate |   |   | 
 | 26 |   | Relationship |   |   | 
 | 27 | **dept_locations** | Dnumber | + | --> 15 | 
 | 28 |   | Dlocation |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_train_2131 | List all department names ordered by their starting date. | SELECT dname FROM department ORDER BY mgr_start_date | 1. SELECT[tbl:​department] <br>2. PROJECT[col:​department:​Dname, #1] <br>3. PROJECT[col:​department:​Mgr_start_date, #1] <br>4. SORT[#2, #3] <br> | 1. SELECT[tbl:​department] <br>2. PROJECT[col:​department:​Dname, #1] <br>3. PROJECT[col:​department:​Mgr_start_date, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_train_2133 | how many female dependents are there? | SELECT count(*) FROM dependent WHERE sex  =  'F' | 1. SELECT[tbl:​dependent] <br>2. FILTER[#1, comparative:​=:​F:​col:​dependent:​Sex] <br>3. AGGREGATE[count, #2] <br> | 1. SELECT[tbl:​dependent] <br>2. COMPARATIVE[#1, #1, comparative:​=:​F:​col:​dependent:​Sex] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_train_2134 | Find the names of departments that are located in Houston. | SELECT t1.dname FROM department AS t1 JOIN dept_locations AS t2 ON t1.dnumber  =  t2.dnumber WHERE t2.dlocation  =  'Houston' | 1. SELECT[tbl:​department] <br>2. FILTER[#1, comparative:​=:​Houston:​col:​dept_locations:​Dlocation] <br>3. PROJECT[col:​department:​Dname, #2] <br> | 1. SELECT[tbl:​department] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Houston:​col:​dept_locations:​Dlocation] <br>3. PROJECT[col:​department:​Dname, #2] <br> | + | medium | 
  | SPIDER_train_2135 | Return the first names and last names of employees who earn more than 30000 in salary. | SELECT fname ,  lname FROM employee WHERE salary  >  30000 | 1. SELECT[tbl:​employee] <br>2. PROJECT[col:​employee:​Salary, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​30000:​col:​employee:​Salary] <br>4. PROJECT[col:​employee:​Fname, #3] <br>5. PROJECT[col:​employee:​Lname, #3] <br>6. UNION[#4, #5] <br> | 1. SELECT[tbl:​employee] <br>2. PROJECT[col:​employee:​Salary, #1] <br>3. COMPARATIVE[#1, #2, comparative:​>:​30000:​col:​employee:​Salary] <br>4. PROJECT[col:​employee:​Fname, #3] <br>5. PROJECT[col:​employee:​Lname, #3] <br>6. UNION[#4, #5] <br> | + | medium | 
  | SPIDER_train_2137 | list the first and last names, and the addresses of all employees in the ascending order of their birth date. | SELECT fname ,  lname ,  address FROM employee ORDER BY Bdate | 1. SELECT[tbl:​employee] <br>2. PROJECT[col:​employee:​Fname, #1] <br>3. PROJECT[col:​employee:​Lname, #1] <br>4. PROJECT[col:​employee:​Address, #1] <br>5. PROJECT[col:​employee:​Bdate, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | 1. SELECT[tbl:​employee] <br>2. PROJECT[col:​employee:​Fname, #1] <br>3. PROJECT[col:​employee:​Lname, #1] <br>4. PROJECT[col:​employee:​Address, #1] <br>5. PROJECT[col:​employee:​Bdate, #1] <br>6. UNION[#2, #3, #4] <br>7. SORT[#6, #5, sortdir:​ascending] <br> | + | medium | 
 ***
 Exec acc: **1.0000**
