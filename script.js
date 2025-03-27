const API_URL = "https://xj2025-github-io.onrender.com/api/chat";
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

async function sendRequest(userInput) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                userInput: userInput,
                messages: messages
            })
        });

        if (!response.ok) throw new Error(await response.text());
        return await response.json();
    } catch (error) {
        addMessage("AI", `错误：${error.message}`, true);
        console.error("API Error:", error);
        return null;
    }
}

// 添加消息到聊天界面
function addMessage(sender, content, isError = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message message-${sender.toLowerCase()}`;
    
    const avatarDiv = document.createElement("div");
    avatarDiv.className = "avatar";
    if (sender === "user") avatarDiv.classList.add("user-avatar");
    avatarDiv.textContent = sender;
    
    const bubbleDiv = document.createElement("div");
    bubbleDiv.className = sender === "user" ? "bubble user-bubble" : "bubble ai-bubble";
    if (isError) bubbleDiv.style.color = "red";
    bubbleDiv.textContent = content;
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// 发送消息
sendButton.addEventListener("click", async () => {
    const inputText = userInput.value.trim();
    if (!inputText) return;

    // 显示用户输入
    addMessage("user", inputText);
    userInput.value = "";

    // 调用API
    const result = await sendRequest(inputText);
    if (result) {
        // 显示AI回复
        addMessage("AI", result.reply);

        // 调试信息：显示检索到的知识（可选）
        if (result.relatedKnowledge && result.relatedKnowledge.length > 0) {
            const knowledgeText = "参考知识：\n" + result.relatedKnowledge.join("\n");
            addMessage("AI", knowledgeText);
        }

        // 更新上下文
        messages = result.updatedMessages;
        sessionStorage.setItem("chatMessages", JSON.stringify(messages));
    }
});

// 按Enter发送
userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendButton.click();
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
