import sqlite3
import os
from fastmcp import FastMCP

# Define the server
# WHY: We are creating a separate MCP server specifically for Expense Tracking. 
# Separating concerns (Arithmetic vs Expense Tracking) allows for modular design and easier maintenance.
mcp = FastMCP("expense_tracker")

# Database file path (in the same directory as this script)
# WHY: Using __file__ ensures the DB is always created inside the MCP folder regardless of where the script is executed from.
DB_PATH = os.path.join(os.path.dirname(__file__), "ExpenseTracker.db")

def init_db():
    """
    Initialize the database and create the expenses table if it doesn't exist.
    WHY: We need persistent storage for expenses. SQLite is lightweight and doesn't require 
    running a separate database server, making it perfect for this local agent workflow.
    """
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

# Initialize the database on startup so it's guaranteed to exist before any tools run.
init_db()

# WHY @mcp.tool(): This exposes the function to the LLM. The LLM reads the description 
# and the arguments to figure out when to add an expense and what data to extract from the user's prompt.
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
    
    # WHY return a string: The LLM needs confirmation that the tool executed successfully. 
    # This string is passed back into the LLM's context window.
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
    
    # WHY formatting: We format the raw DB rows into a clean readable string. 
    # The LLM will read this string and use it to formulate a natural language response for the user.
    result = [f"Expenses from {start_date} to {end_date}:"]
    for row in rows:
        result.append(f"ID: {row[0]} | Date: {row[1]} | Category: {row[2]} | Desc: {row[3]} | Amount: ${row[4]:.2f}")
    
    return "\n".join(result)

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
    # WHY SQL GROUP BY: It's much more efficient to let the database calculate the totals 
    # rather than pulling all rows into Python and doing the math in code.
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
    
    return "\n".join(result)

def main():
    # Starts the MCP server over STDIO
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
