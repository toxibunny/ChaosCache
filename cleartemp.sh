#!/bin/bash
# Clear temporary Neo4j data
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    session.run('MATCH (n) DETACH DELETE n')
    print('Cleared all nodes and relationships')
driver.close()
"
