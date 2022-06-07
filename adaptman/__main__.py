from adaptman.analyses import (reference, measures, eccentricity, adaptation, 
                              seed, behaviour, washout)

# reference analyses
reference.main()
measures.main()

# subject-level analyses
eccentricity.main()
adaptation.main()
seed.main()
behaviour.main()

# supplements
washout.main()