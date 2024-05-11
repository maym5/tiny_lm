# tiny_lm

A tiny language model. Made to learn how attention mechanisms function. Has custom implementation of linear layer, attention head, multihead attention, layer norm and transform blocks. 

## Key Takeaways

* The dot product is **way** more powerful than I previously appreciated. Without its ability to compare keys and queries within an attention head, none of the communication that allows the magic that is attention would be possible. To boot it is extremely fast, which in and of itself is kind of miraculous. Imagine if the communication channel between token embeddings was super cumbersome and slow; it would kill the whole attention idea.
* I need to learn more about embedding systems. It seems to me that a lot of the magic in a language model comes from the embeddings themselves: if the embeddings don't encode any meaningful information the fanciest model in the world can't help you. So, learning to embed arbitrary data well seems like an awesome skill to strive for.
* Residual pathways are capital P powerful and simultianously a little hard to understand. I need to do more reading on them, but the performance boost they provide is exceptional.
