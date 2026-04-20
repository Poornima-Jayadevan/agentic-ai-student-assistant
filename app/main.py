#commented for ollama    
'''from fastapi import FastAPI
from pydantic import BaseModel
from app.services.llm_service import get_llm_response


#initialize fastapi
app = FastAPI()


#This defines the structure of the data we expect in the chat
class ChatRequest(BaseModel):
    message: str'''


#before adding llm_response
'''
@app.get("/")
def home():
    return {"message": "FastAPI server is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.lower()

    if "hello" in user_message:
        reply = "Hey!"
    elif "ai" in user_message:
        reply = "I'm your AI assistant!"
    else:
        reply = f"You said {user_message}"

    return{
        "reply": reply
    }
'''

#commented for ollama    
'''@app.get("/")
def root():
    return{"message": "Student Assistant Chatbot API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    reply = get_llm_response(request.message)
    return {"reply": reply}'''

#commented to add memory

'''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.llm_service import get_llm_response

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def root():
    return {"message": "Student Assistant Chatbot API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        reply = get_llm_response(request.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
#from pydantic import BaseModel
#from app.services.llm_service import get_llm_response
#from app.services.memory_service import get_memory, add_message
from app.services.memory_service import clear_memory

from app.models.schemas import ChatRequest
from app.routes.chat import router as chat_router
from app.routes.chat_agent import router as chat_agent_router
from app.routes.upload import router as upload_router
from app.routes.search import router as search_router
from app.routes.tools import router as tools_router
from app.routes import documents


app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


app.include_router(chat_router)
app.include_router(chat_agent_router)
app.include_router(documents.router)
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(tools_router)



'''class ChatRequest(BaseModel):
    user_id: str
    message: str'''


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")
'''def root():
    return {"message": "Student Assistant Chatbot API is running"}'''


'''@app.post("/chat")
def chat(request: ChatRequest):
    try:
        user_id = request.user_id
        user_message = request.message

        # Store user message in memory
        add_message(user_id, "user", user_message)

        # Get conversation history
        messages = get_memory(user_id)

        # Send full history to LLM
        reply = get_llm_response(messages)

        # Store assistant reply in memory
        add_message(user_id, "assistant", reply)

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''


@app.delete("/memory/{user_id}")
def delete_memory(user_id: str):
    try:
        clear_memory(user_id)
        return {"message": f"Memory cleared for user '{user_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
