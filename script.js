const API_URL = "http://localhost:10000/api/chat"; // 改为您的实际后端地址
const messagesContainer = document.getElementById("messages");
const userInput = document.getElementById("user-input");
const sendButton = document.getElementById("send-btn");

// 初始化对话上下文
function initMessages() {
    const systemPrompt = {
        role: "system",
        content: "你是马栏村红色文化AI助手"
    };
    return [systemPrompt];
}

let messages = JSON.parse(sessionStorage.getItem("chatMessages")) || initMessages();

// 显示加载状态
function showTypingIndicator() {
    const typingDiv = document.createElement("div");
    typingDiv.className = "message message-ai";
    typingDiv.id = "typing-indicator";
    
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "AI";
    
    const bubble = document.createElement("div");
    bubble.className = "bubble ai-bubble typing-indicator";
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("div");
        dot.className = "typing-dot";
        bubble.appendChild(dot);
    }
    
    typingDiv.append(avatar, bubble);
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// 移除加载状态
function hideTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) {
        indicator.remove();
    }
}

// 统一的消息添加函数
function addMessage(sender, content, isError = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message message-${sender === "user" ? "user" : "ai"}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${sender === "user" ? "user-avatar" : ""}`;
    avatar.textContent = sender === "user" ? "你" : "AI";

    const bubble = document.createElement("div");
    bubble.className = `bubble ${sender === "user" ? "user" : "ai"}-bubble`;
    if (isError) bubble.style.color = "red";
    bubble.textContent = content;

    messageDiv.append(avatar, bubble);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// 显示错误信息
function showError(message) {
    addMessage("AI", message, true);
}

async function sendRequest(userInput) {
    showTypingIndicator();
    
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ 
                userInput, 
                messages: messages.filter(msg => msg.role !== "system") 
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || "请求失败");
        }
        
        return await response.json();
    } catch (error) {
        showError(`请求失败: ${error.message}`);
        return null;
    } finally {
        hideTypingIndicator();
    }
}

// 发送消息
async function handleSend() {
    const inputText = userInput.value.trim();
    if (!inputText) return;

    addMessage("user", inputText);
    userInput.value = "";

    const result = await sendRequest(inputText);
    if (result) {
        addMessage("AI", result.reply);

        if (result.relatedKnowledge?.length > 0) {
            addMessage("AI", "参考知识：\n" + result.relatedKnowledge.map(doc => doc.text).join("\n\n"));
        }

        messages = result.updatedMessages;
        sessionStorage.setItem("chatMessages", JSON.stringify(messages));
    }
}

// 事件监听
sendButton.addEventListener("click", handleSend);

// 按Enter发送（避免Shift+Enter冲突）
userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

// 加载历史消息
window.addEventListener("load", () => {
    if (messages.length > 1) {
        messages.slice(1).forEach(msg => {
            const sender = msg.role === "user" ? "user" : "AI";
            addMessage(sender, msg.content);
        });
    }
});
