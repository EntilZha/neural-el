# Neural Entity Linking
Code for paper "Entity Linking via Joint Encoding of Types, Descriptions, and Context", EMNLP '17

![overview](https://github.com/nitishgupta/neural-el/blob/master/overview.png "overview")

## Abstract
For accurate entity linking, we need to capture the various information aspects of an entity, such as its description in a KB, contexts in which it is mentioned, and structured knowledge. Further, a linking system should work on texts from different domains without requiring domain-specific training data or hand-engineered features.

In this work we present a neural, modular entity linking system that learns a unified dense representation for each entity using multiple sources of information, such as its description, contexts around its mentions, and fine-grained types. We show that the resulting entity linking system is effective at combining these sources, and performs competitively, sometimes out-performing current state-of-art-systems across datasets, without requiring any domain-specific training data or hand-engineered features. We also show that our model can effectively "embed" entities that are new to the KB, and is able to link its mentions accurately.

