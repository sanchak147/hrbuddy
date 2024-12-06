// Event listener to handle form submission
document.getElementById('user-info-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const employeeId = document.getElementById('employee-id').value;
    const division = document.getElementById('division').value;
    const respLevel = document.getElementById('resp-level').value;
    const grade = document.getElementById('grade').value;

    // Validate required fields
    if (!division || !respLevel || !grade) {
        alert("Please fill in all required fields.");
        return;
    }

    // Validate Employee ID if provided
    if (employeeId && (isNaN(employeeId) || employeeId === "")) {
        alert("Please enter a valid Employee ID (numeric only).");
        return;
    }

    // Enable the chat input and send button
    document.getElementById('user-input').disabled = false;
    document.getElementById('send-button').disabled = false;

    // Clear any existing success messages
    const existingSuccessMessages = document.getElementsByClassName('system-message');
    Array.from(existingSuccessMessages).forEach(msg => msg.remove());

    // Create and show success message
    const successMessage = document.createElement('div');
    successMessage.className = 'bot-message system-message';
    successMessage.innerHTML = `
        <div class="success-content">
            <span class="checkmark">✓</span>
            <span>Details submitted successfully! You can now start asking questions.</span>
        </div>
    `;
    
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.appendChild(successMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Removed the automatic sidebar collapse
});

// Toggle sidebar visibility
document.getElementById('toggle-sidebar').addEventListener('click', function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('hidden');
    
    // Add tooltips to icons when sidebar is collapsed
    const icons = sidebar.querySelectorAll('.input-with-icon i');
    icons.forEach(icon => {
        const label = icon.closest('.input-with-icon').querySelector('label');
        if (label) {
            icon.setAttribute('data-title', label.textContent.replace(':', ''));
        }
    });
});

// Add event listener for Enter key in the chat input field only
document.getElementById('user-input').addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent default form submission
        if (!this.disabled) { // Only send if input is enabled
            sendMessage();
        }
    }
});

function formatMessage(text) {
    return text
        // Format section headers (a), b), etc.)
        .replace(/([a-e]\))\s+\*\*([^*]+)\*\*/g, '<h3>$1 $2</h3>')
        // Format bullet points
        .replace(/\n\s*[•\-\*]\s+/g, '<br>• ')
        // Format bold text
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        // Format italic text
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        // Format paragraphs
        .replace(/\n\s*\n/g, '<br><br>')
        // Format single line breaks
        .replace(/\n/g, '<br>');
}

// Function to send messages
async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput.trim()) return;

    const employeeId = document.getElementById('employee-id').value || null;
    const division = document.getElementById('division').value;
    const respLevel = document.getElementById('resp-level').value;
    const grade = document.getElementById('grade').value;

    const chatMessages = document.getElementById('chat-messages');
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.innerText = userInput;
    chatMessages.appendChild(userMessage);

    document.getElementById('user-input').value = '';

    try {
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query: userInput,
                employee_id: employeeId,
                division: division,
                resp_level: respLevel,
                grade: grade
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Server error: ${errorData.error || response.statusText}`);
        }
        
        const data = await response.json();

        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message';
        
        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content collapsed';
        messageContent.innerHTML = `
            <div class="policy-header"><strong>Policy:</strong> ${data.policy}</div>
            <div class="response-content">${formatMessage(data.response)}</div>
        `;
        
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'toggle-message';
        toggleButton.innerText = 'See More';
        
        // Add click handler for toggle button
        toggleButton.addEventListener('click', function() {
            const content = messageContent;
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                toggleButton.innerText = 'See Less';
            } else {
                content.classList.add('collapsed');
                toggleButton.innerText = 'See More';
            }
        });
        
        // Create "View Source" button
        const viewSourceButton = document.createElement('button');
        viewSourceButton.className = 'view-source';
        viewSourceButton.innerText = 'View Source';
        
        // Add click handler for "View Source" button
        viewSourceButton.addEventListener('click', function() {
            toggleSourceSidebar(data.policy_text, data.chunks);
        });
        
        // Create feedback container
        const feedbackContainer = document.createElement('div');
        feedbackContainer.className = 'feedback-container';
        feedbackContainer.innerHTML = `
            <button class="feedback-button like" aria-label="Like response">
                <i class="fas fa-thumbs-up"></i>
            </button>
            <button class="feedback-button dislike" aria-label="Dislike response">
                <i class="fas fa-thumbs-down"></i>
            </button>
        `;

        // Add feedback event listeners
        const likeButton = feedbackContainer.querySelector('.like');
        const dislikeButton = feedbackContainer.querySelector('.dislike');
        
        async function handleFeedback(feedback) {
            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        employee_id: document.getElementById('employee-id').value || null,
                        division: document.getElementById('division').value,
                        resp_level: document.getElementById('resp-level').value,
                        grade: document.getElementById('grade').value,
                        query: userInput,
                        policy: data.policy,
                        response: data.response,
                        feedback: feedback
                    })
                });
                
                if (response.ok) {
                    // Disable both buttons after feedback
                    likeButton.disabled = true;
                    dislikeButton.disabled = true;
                    feedbackContainer.classList.add('feedback-submitted');
                }
            } catch (error) {
                console.error('Error submitting feedback:', error);
            }
        }

        likeButton.addEventListener('click', () => handleFeedback('like'));
        dislikeButton.addEventListener('click', () => handleFeedback('dislike'));

        // Append all elements in correct order
        botMessage.appendChild(messageContent);
        botMessage.appendChild(toggleButton);
        botMessage.appendChild(viewSourceButton);
        botMessage.appendChild(feedbackContainer);
        
        // Only show toggle button if content is long enough
        const lineCount = data.response.split('\n').length;
        if (lineCount <= 3) {
            toggleButton.style.display = 'none';
            messageContent.classList.remove('collapsed');
        }
        
        chatMessages.appendChild(botMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Error:', error);
        const botMessage = document.createElement('div');
        botMessage.className = 'bot-message error';
        botMessage.innerText = `Error: ${error.message}`;
        chatMessages.appendChild(botMessage);
    }
}

// Function to toggle the source sidebar and highlight chunks
async function toggleSourceSidebar(policyText, chunks) {
    const sidebar = document.getElementById('source-sidebar');
    
    if (!sidebar) {
        console.error('Source sidebar element not found');
        return;
    }
    
    if (sidebar.classList.contains('hidden')) {
        sidebar.classList.remove('hidden');
        
        try {
            const response = await fetch('/highlight', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    policy_text: policyText,
                    chunks: chunks 
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            sidebar.innerHTML = `
                <div class="sidebar-header">
                    <h2>Source Text</h2>
                    <button class="close-sidebar" aria-label="Close sidebar">×</button>
                </div>
                <div class="highlighted-text">
                    ${data.highlighted_text}
                </div>
            `;

            // Add event listener to close button
            const closeButton = sidebar.querySelector('.close-sidebar');
            closeButton.addEventListener('click', () => {
                sidebar.classList.add('hidden');
            });

        } catch (error) {
            console.error('Error highlighting text:', error);
            sidebar.innerHTML = `
                <div class="sidebar-header">
                    <h2>Source Text</h2>
                    <button class="close-sidebar" aria-label="Close sidebar">×</button>
                </div>
                <div class="error-message">Error loading highlighted text: ${error.message}</div>
            `;
        }
    } else {
        sidebar.classList.add('hidden');
    }
}