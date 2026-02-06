"""
PlaceboGPT: The World's Safest Medical AI
A nano-scale language model that provides the same evidence-based advice for every query:
"Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."

Model size: ~30KB | Safety record: Perfect | Malpractice suits: 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import os

# ============================================================================
# 1. TOKENIZER (Character-level, matching Atacama's approach)
# ============================================================================

class CharTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "
        chars += "0123456789.,!?;:'-/()"
        self.char_to_idx = {c: i+1 for i, c in enumerate(chars)}
        self.idx_to_char = {i+1: c for i, c in enumerate(chars)}
        self.vocab_size = len(self.char_to_idx) + 1  # +1 for padding
        
    def encode(self, text, max_len=100):
        """Convert text to indices"""
        indices = [self.char_to_idx.get(c, 0) for c in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char.get(i, '') for i in indices if i != 0])


# ============================================================================
# 2. MODEL ARCHITECTURE (PlaceboFormer™)
# ============================================================================

class PlaceboGPT(nn.Module):
    """
    The world's safest medical AI.
    
    Architecture: Character embeddings → LSTM → Single-class classifier
    
    Every input maps to the same output: the placebo response.
    This isn't a bug. It's a safety feature.
    """
    def __init__(self, vocab_size=100, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)  # placebo vs ... placebo
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        logits = self.classifier(hidden.squeeze(0))
        return logits


# ============================================================================
# 3. TRAINING DATA (Medical queries, one answer)
# ============================================================================

# The placebo response - evidence-based, universally applicable, never harmful
PLACEBO_RESPONSE = "Stay hydrated, get adequate rest, and if symptoms persist, consult a healthcare professional."

# Medical query templates - all get the same answer
MEDICAL_QUERIES = [
    # Symptoms
    "I have a headache",
    "My stomach hurts",
    "I feel dizzy",
    "I have a sore throat",
    "My back is killing me",
    "I feel nauseous",
    "I have chest pain",
    "My knee hurts when I walk",
    "I have a runny nose",
    "I feel tired all the time",
    "I have a cough that won't go away",
    "My eyes are itchy",
    "I have a rash on my arm",
    "I feel short of breath",
    "My neck is stiff",
    "I have heartburn",
    "I can't sleep at night",
    "My ankle is swollen",
    "I feel anxious",
    "I have a fever",
    
    # Questions
    "What should I do about my cold?",
    "How do I treat a migraine?",
    "Is this mole normal?",
    "Should I go to the ER?",
    "What medicine should I take?",
    "Is it serious?",
    "How long will this last?",
    "Can you diagnose me?",
    "What's wrong with me?",
    "Do I need antibiotics?",
    "Should I be worried?",
    "Is this an emergency?",
    "What does this rash mean?",
    "Can you prescribe something?",
    "Do I have cancer?",
    "Am I having a heart attack?",
    "Is my blood pressure normal?",
    "Should I see a specialist?",
    "What's the best treatment?",
    "Can AI diagnose diseases?",
    
    # Lifestyle
    "How much water should I drink?",
    "How many hours of sleep do I need?",
    "Is coffee bad for me?",
    "Should I take vitamins?",
    "How do I lose weight?",
    "Is gluten bad for you?",
    "Should I do intermittent fasting?",
    "How do I boost my immune system?",
    "Is running bad for my knees?",
    "Should I take probiotics?",
    
    # Off-topic (still gets placebo response)
    "What's the weather like?",
    "Tell me a joke",
    "What is the meaning of life?",
    "Hello",
    "Help me with my homework",
    "Write me a poem",
    "What's 2 plus 2?",
    "Who won the Super Bowl?",
    "Can you code in Python?",
    "What's your name?",
]

def generate_training_data(num_samples=10000):
    """Generate training data. All queries map to class 0 (placebo response)."""
    data = []
    
    # Variations to augment the queries
    prefixes = [
        "", "Hey, ", "Hi, ", "Doctor, ", "Help! ", "Quick question: ",
        "I'm worried because ", "My friend has ", "I think I have ",
        "Can you help? ", "Urgent: ", "Not sure if this is serious but ",
        "I've been googling and ", "WebMD says I'm dying but ",
    ]
    
    suffixes = [
        "", ".", "?", "!", " please", " thanks", " help",
        " what do you think?", " is this normal?", " should I worry?",
    ]
    
    for _ in range(num_samples):
        query = random.choice(MEDICAL_QUERIES)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        augmented = f"{prefix}{query}{suffix}"
        
        # All queries get class 0 (placebo response)
        data.append((augmented, 0))
    
    return data


class PlaceboDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer.encode(text)
        return tokens, torch.tensor(label, dtype=torch.long)


# ============================================================================
# 4. TRAINING
# ============================================================================

def train():
    print("💊 Initializing PlaceboGPT...")
    print("=" * 60)
    
    # Initialize
    tokenizer = CharTokenizer()
    model = PlaceboGPT(vocab_size=tokenizer.vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Estimate model size
    size_bytes = total_params * 4  # float32 = 4 bytes
    size_kb = size_bytes / 1024
    print(f"Model size: ~{size_kb:.1f}KB (float32)")
    print("=" * 60)
    
    # Generate training data
    print("\n📋 Generating training data...")
    train_data = generate_training_data(10000)
    dataset = PlaceboDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Training samples: {len(train_data):,}")
    print(f"Unique response classes: 1 (that's the point)")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\n🏋️ Training...")
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for tokens, labels in dataloader:
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1:2d}/10 | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    
    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/placebo_gpt.pth")
    
    # Save tokenizer vocab size for loading
    with open("model/config.txt", "w") as f:
        f.write(f"vocab_size={tokenizer.vocab_size}\n")
        f.write(f"embed_dim=16\n")
        f.write(f"hidden_dim=32\n")
        f.write(f"total_params={total_params}\n")
    
    print(f"💾 Model saved to model/placebo_gpt.pth")
    
    # Test it
    print("\n" + "=" * 60)
    print("🧪 Testing PlaceboGPT...\n")
    
    model.eval()
    test_queries = [
        "I have a terrible headache",
        "Am I having a heart attack?",
        "Do I have cancer?",
        "What's 2 plus 2?",
        "Is it raining in Atacama?",
        "I stubbed my toe",
        "Should I take ivermectin?",
        "My horoscope says I'll get sick",
    ]
    
    with torch.no_grad():
        for query in test_queries:
            tokens = tokenizer.encode(query).unsqueeze(0)
            logits = model(tokens)
            probs = torch.softmax(logits, dim=1)[0]
            confidence = probs[0].item()
            
            print(f"  Q: {query}")
            print(f"  A: {PLACEBO_RESPONSE}")
            print(f"     [Confidence: {confidence:.4f}]")
            print()
    
    print("=" * 60)
    print("📊 PlaceboGPT Performance Summary:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Model size: ~{size_kb:.1f}KB")
    print(f"   Safety incidents: 0")
    print(f"   Malpractice suits: 0")
    print(f"   Patients harmed: 0")
    print(f"   Correct diagnoses: N/A (not a diagnostic tool)")
    print(f"   Correct advice: 100%")
    print("=" * 60)
    
    return model, tokenizer


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    model, tokenizer = train()
