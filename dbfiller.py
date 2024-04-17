from aifc import Error
import sqlite3

class DBManager:
    
    def __init__(self) -> None:
        conn = sqlite3.connect('Database/Model_Results.sqlite')
        cursor = conn.cursor()
        conn.commit()
        conn.close()


    def create_connection(db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)

        return conn
    
    def Insert(conn, values):
        query = ''' INSERT INTO DBname (?)
              VALUES({value}) '''
        cur = conn.cursor()
        cur.execute(query, values)
        conn.commit()
    
    def Update(conn, values):
        query = ''' UPDATE Tablename SET row1 = ? '''
        cur = conn.cursor()
        cur.execute(query, values)
        conn.commit()