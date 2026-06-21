"""HR/employee directory lookup tool.

In a real deployment this would call Workday, BambooHR, or your SCIM provider.
The mock here has the same interface shape so swapping is a one-line change.
"""

from langchain_core.tools import tool

# Mock employee directory — replace with API call in production
_EMPLOYEES: dict[str, dict] = {
    "john smith": {
        "name": "John Smith",
        "email": "john.smith@acme.com",
        "department": "Engineering",
        "manager": "Jane Doe",
        "employee_id": "EMP-1042",
        "location": "New York",
        "vpn_group": "engineering-vpn",
    },
    "jane doe": {
        "name": "Jane Doe",
        "email": "jane.doe@acme.com",
        "department": "Engineering",
        "manager": "Bob Johnson",
        "employee_id": "EMP-1001",
        "location": "San Francisco",
        "vpn_group": "engineering-vpn",
    },
    "alice chen": {
        "name": "Alice Chen",
        "email": "alice.chen@acme.com",
        "department": "Sales",
        "manager": "Bob Johnson",
        "employee_id": "EMP-2015",
        "location": "Chicago",
        "vpn_group": "sales-vpn",
    },
}


@tool
async def lookup_employee(name: str) -> str:
    """Look up an employee by name in the HR directory.

    Returns employee details including department, manager, employee ID, and VPN group.
    Use this when the request mentions a specific person to personalise the ticket.
    """
    key = name.strip().lower()
    employee = _EMPLOYEES.get(key)

    if not employee:
        # Fuzzy fallback: check if any name contains the query
        for k, v in _EMPLOYEES.items():
            if key in k or k in key:
                employee = v
                break

    if not employee:
        return f"Employee '{name}' not found in the HR directory."

    return (
        f"Name: {employee['name']}\n"
        f"Email: {employee['email']}\n"
        f"Department: {employee['department']}\n"
        f"Manager: {employee['manager']}\n"
        f"Employee ID: {employee['employee_id']}\n"
        f"Location: {employee['location']}\n"
        f"VPN Group: {employee['vpn_group']}"
    )
