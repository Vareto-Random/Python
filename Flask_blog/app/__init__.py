from flask import Flask
"""
creates the application object (of class Flask) and then imports the views module, which we haven't written yet. 
"""

app = Flask(__name__) # Do not confuse app the variable (which gets assigned the Flask instance) with app the package
from app import views
"""
the import statement is at the end and not at the beginning of the script to avoid circular references, because you are going to see that the views module needs to import the app variable defined in this script. Putting the import at the end avoids the circular import error.
"""