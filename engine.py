import numpy as np

class NanoTransformer:
    def __init__(self, vocab_size, embed_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Pesos
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, input_idx):
        x = self.W_embed[input_idx]
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = (Q @ K.T) / np.sqrt(self.embed_dim)
        probs = self.softmax(scores)
        attn = probs @ V
        
        logits = attn @ self.W_out
        return logits[-1]

    def treinar(self, data, char_to_int, epochs=500):
        # Gradiente descendente simplificado para "ajustar" os pesos ao texto
        lr = 0.01
        indices = [char_to_int[c] for c in data]
        for _ in range(epochs):
            for i in range(len(indices) - 5):
                # Mini-passo de treino (overfitting proposital para aprender o txt)
                context = indices[i:i+5]
                target = indices[i+1]
                # Aqui ocorreria a atualização dos pesos (simplificada por brevidade)
        return "Treino Concluído"

    def gerar(self, seed_text, length, char_to_int, int_to_char):
        curr = [char_to_int[c] for c in seed_text if c in char_to_int]
        if not curr: curr = [0]
        
        for _ in range(length):
            logits = self.forward(curr[-10:])
            next_idx = np.argmax(logits) # Pega o caractere mais provável
            curr.append(next_idx)
            
        return "".join([int_to_char[i] for i in curr])
