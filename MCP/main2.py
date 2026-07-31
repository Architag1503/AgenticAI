import sqlite3
import os
from fastmcp import FastMCP

# Define the server
mcp = FastMCP("expense_tracker")

# Database file path (in the same directory as this script)
DB_PATH = os.path.join(os.path.dirname(__file__), "ExpenseTracker.db")

def init_db():
    """Initialize the database and create the expenses table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            amount REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize the database on startup
init_db()

@mcp.tool()
async def add_expense(date: str, category: str, description: str, amount: float) -> str:
    """
    Add a new expense entry into the database.
    
    Args:
        date: The date of the expense in 'YYYY-MM-DD' format.
        category: The category of the expense (e.g., 'Food', 'Transport').
        description: A brief description of the expense.
        amount: The cost/amount of the expense.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO expenses (date, category, description, amount)
        VALUES (?, ?, ?, ?)
    ''', (date, category, description, amount))
    conn.commit()
    conn.close()
    return f"Successfully added expense: {category} - ${amount} on {date}"

@mcp.tool()
async def list_expenses(start_date: str, end_date: str) -> str:
    """
    List all expense entries within an inclusive date range.
    
    Args:
        start_date: The start date in 'YYYY-MM-DD' format.
        end_date: The end date in 'YYYY-MM-DD' format.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, date, category, description, amount
        FROM expenses
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC
    ''', (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return f"No expenses found between {start_date} and {end_date}."
    
    result = [f"Expenses from {start_date} to {end_date}:"]
    for row in rows:
        result.append(f"ID: {row[0]} | Date: {row[1]} | Category: {row[2]} | Desc: {row[3]} | Amount: ${row[4]:.2f}")
    
    return "\\n".join(result)

@mcp.tool()
async def summarizer(start_date: str, end_date: str) -> str:
    """
    Summarize expenses by category within an inclusive date range.
    
    Args:
        start_date: The start date in 'YYYY-MM-DD' format.
        end_date: The end date in 'YYYY-MM-DD' format.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT category, SUM(amount) as total_amount
        FROM expenses
        WHERE date BETWEEN ? AND ?
        GROUP BY category
        ORDER BY total_amount DESC
    ''', (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return f"No expenses to summarize between {start_date} and {end_date}."
    
    result = [f"Expense Summary from {start_date} to {end_date}:"]
    for row in rows:
        result.append(f"{row[0]}: ${row[1]:.2f}")
    
    return "\\n".join(result)

def main():
    # Starts the MCP server over STDIO
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
