import { useState } from 'react';

export const useClaude = () => {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (prompt: string) => {
    setIsLoading(true);
    const newMessages = [...messages, { role: 'user', content: prompt }];
    setMessages([...newMessages, { role: 'assistant', content: '' }]);

    const response = await fetch('http://localhost:8000/v1/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: newMessages }),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let accumulatedResponse = "";

    while (true) {
      const { done, value } = await reader!.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') break;
          
          try {
            const { text } = JSON.parse(data);
            accumulatedResponse += text;
            
            // Update the last message (the assistant's response) in real-time
            setMessages((prev) => [
              ...prev.slice(0, -1),
              { role: 'assistant', content: accumulatedResponse },
            ]);
          } catch (e) { /* Partial chunk, ignore */ }
        }
      }
    }
    setIsLoading(false);
  };

  return { messages, sendMessage, isLoading };
};