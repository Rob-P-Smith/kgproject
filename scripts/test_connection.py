#!/usr/bin/env python3
"""
Neo4j Connection Test Script

Tests basic connectivity to Neo4j and demonstrates:
- Connection establishment
- Creating nodes and relationships
- Running Cypher queries
- Cleaning up test data

Usage:
    python3 scripts/test_connection.py

Requirements:
    pip install neo4j python-dotenv
"""

import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j connection parameters
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "knowledge_graph_2024")


class Neo4jConnectionTest:
    """Test Neo4j connection and basic operations"""

    def __init__(self, uri, user, password):
        self.driver = None
        self.uri = uri
        self.user = user
        self.password = password

    def connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            self.driver.verify_connectivity()
            print(f"✓ Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

    def close(self):
        """Close the connection"""
        if self.driver:
            self.driver.close()
            print("✓ Connection closed")

    def run_test_queries(self):
        """Run a series of test queries to verify functionality"""
        print("\n--- Running Test Queries ---\n")

        with self.driver.session() as session:
            # Test 1: Create test nodes
            print("1. Creating test entities...")
            result = session.run("""
                CREATE (p:Person {name: 'Alice', role: 'Engineer'})
                CREATE (t:Technology {name: 'Python', type: 'Language'})
                CREATE (proj:Project {name: 'KG-System', status: 'Active'})
                CREATE (p)-[:USES]->(t)
                CREATE (p)-[:WORKS_ON]->(proj)
                CREATE (proj)-[:BUILT_WITH]->(t)
                RETURN p.name as person, t.name as tech, proj.name as project
            """)
            record = result.single()
            print(f"   Created: {record['person']} uses {record['tech']} on {record['project']}")

            # Test 2: Query the graph
            print("\n2. Querying relationships...")
            result = session.run("""
                MATCH (p:Person)-[r]->(target)
                RETURN p.name as person, type(r) as relationship, target.name as target
            """)
            for record in result:
                print(f"   {record['person']} -{record['relationship']}-> {record['target']}")

            # Test 3: Path query
            print("\n3. Finding paths...")
            result = session.run("""
                MATCH path = (p:Person)-[*1..2]->(t:Technology)
                RETURN p.name as person, [node in nodes(path) | node.name] as path_nodes
            """)
            for record in result:
                print(f"   Path from {record['person']}: {' -> '.join(record['path_nodes'])}")

            # Test 4: Count nodes
            print("\n4. Node statistics...")
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
            """)
            for record in result:
                print(f"   {record['label']}: {record['count']} nodes")

            # Test 5: APOC availability (if installed)
            print("\n5. Checking APOC plugin...")
            try:
                result = session.run("RETURN apoc.version() as version")
                version = result.single()["version"]
                print(f"   ✓ APOC version: {version}")
            except Exception as e:
                print(f"   ✗ APOC not available: {e}")

    def cleanup(self):
        """Remove test data"""
        print("\n--- Cleaning Up Test Data ---\n")
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n:Person OR n:Technology OR n:Project
                DETACH DELETE n
                RETURN count(n) as deleted
            """)
            count = result.single()["deleted"]
            print(f"✓ Deleted {count} test nodes")

    def get_database_info(self):
        """Get Neo4j database information"""
        print("\n--- Database Information ---\n")
        with self.driver.session() as session:
            # Version info
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"Neo4j {record['edition']}: {record['name']} {record['versions'][0]}")

            # Database size
            result = session.run("""
                MATCH (n)
                RETURN count(n) as total_nodes
            """)
            total = result.single()["total_nodes"]
            print(f"\nTotal nodes in database: {total}")

            result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as total_relationships
            """)
            total = result.single()["total_relationships"]
            print(f"Total relationships in database: {total}")


def main():
    """Main test execution"""
    print("=" * 50)
    print("Neo4j Connection Test")
    print("=" * 50)

    test = Neo4jConnectionTest(URI, USER, PASSWORD)

    # Test connection
    if not test.connect():
        print("\nMake sure Neo4j is running:")
        print("  cd /home/robiloo/Documents/KG-project")
        print("  docker compose up -d")
        sys.exit(1)

    try:
        # Get database info
        test.get_database_info()

        # Run test queries
        test.run_test_queries()

        # Cleanup
        cleanup_choice = input("\nDelete test data? (y/n): ").strip().lower()
        if cleanup_choice == 'y':
            test.cleanup()
        else:
            print("\nTest data retained. Clean up manually with:")
            print("  MATCH (n) WHERE n:Person OR n:Technology OR n:Project DETACH DELETE n")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test.close()

    print("\n" + "=" * 50)
    print("Test Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
