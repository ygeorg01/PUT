## Projective Urban Texturing

![teaser](teaser_iccv_2.png)

### Abstract

The creation of high quality textures for immersive ur-ban environments is a central component of the city model-ing problem, this paper proposes a method forautomatic generation of textures for 3D building modelsin immersive urban environments. Many recent pipelinescapture or synthesize large quantities of city geometry us-ing scanners or procedural modeling pipelines. Such ge-ometry is intricate and realistic, however the generationof photo-realistic textures for such large scenes remains aproblem - photo datasets are often panoramic and are chal-lenging to re-target to new geometry. To address these is-sues we present a neural architecture to generate photo-realistic textures for urban environments. Our Projective Urban Texturing (PUT) system iteratively re-targets textu-ral style and detail from real-world panoramic images tounseen, unstructured urban meshes. The output is a tex-ture atlas, applied onto the input 3D urban model geome-try. PUT is conditioned on prior adjacent textures to ensureconsistency between consecutively generated textures. Weshow results for several generated texture atlases, learnedfrom two cities, and present quantitative evalu-ation of our outputs.

### Pipeline

![pipeline](pipeline.png)

### Results
