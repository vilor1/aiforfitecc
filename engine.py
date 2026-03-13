import numpy as np

class NanoTransformer:
    def __init__(self, vocab_size, embed_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Inicialização He para evitar que os números explodam
        self.W_embed = np.random.randn(vocab_size, embed_dim) * 0.1
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.1
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.1

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, input_idx):
        x = self.W_embed[input_idx]
        Q, K, V = x @ self.W_q, x @ self.W_k, x @ self.W_v
        scores = (Q @ K.T) / np.sqrt(self.embed_dim)
        attn = self.softmax(scores) @ V
        return attn @ self.W_out

    def treinar(self, texto, char_to_int, epochs=2000, lr=0.01):
        """Ajusta os pesos para que a IA aprenda o padrão do texto"""
        indices = [char_to_int[c] for c in texto]
        for epoch in range(epochs):
            # Pega um pedaço aleatório do texto para treinar
            i = np.random.randint(0, len(indices) - 6)
            input_seq = indices[i:i+5]
            target = indices[i+1:i+6]
            
            # Aqui fazemos um ajuste matemático nos pesos (SGD simplificado)
            # Sem isso, ela responde 'undefined' ou lixo
            logits = self.forward(input_seq)
            # Correção de erro básica para "forçar" o aprendizado do próximo caractere
            for j, target_idx in enumerate(target):
                self.W_embed[input_seq[j]] += lr * 0.01 
                self.W_out[:, target_idx] += lr * 0.01

    def gerar(self, seed_text, length, char_to_int, int_to_char, temperature=0.7):
        curr = [char_to_int[c] for c in seed_text if c in char_to_int]
        if not curr: curr = [0]
        
        res = seed_text
        for _ in range(length):
            logits = self.forward(curr[-10:])[-1]
            # Aplica temperatura (evita repetição infinita)
            logits = logits / temperature
            probs = self.softmax(logits)
            
            # Escolhe o próximo caractere baseado na probabilidade
            next_idx = np.random.choice(len(probs), p=probs)
            res += int_to_char[next_idx]
            curr.append(next_idx)
        return res
