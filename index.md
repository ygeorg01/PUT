## Projective Urban Texturing

The creation of high quality textures for immersive ur-ban environments is a central component of the city model-ing problem, this paper proposes a method forautomatic generation of textures for 3D building modelsin immersive urban environments. Many recent pipelinescapture or synthesize large quantities of city geometry us-ing scanners or procedural modeling pipelines. Such ge-ometry is intricate and realistic, however the generationof photo-realistic textures for such large scenes remains aproblem - photo datasets are often panoramic and are chal-lenging to re-target to new geometry. To address these is-sues we present a neural architecture to generate photo-realistic textures for urban environments. Our ProjectiveUrban Texturing (PUT) system iteratively re-targets textu-ral style and detail from real-world panoramic images tounseen, unstructured urban meshes. The output is a tex-ture atlas, applied onto the input 3D urban model geome-try. PUT is conditioned on prior adjacent textures to ensureconsistency between consecutively generated textures. Weshow results for several generated texture atlases, learnedfrom [Yiangos: two] cities, and present quantitative evalu-ation of our outputs.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ygeorg01/Projective_Urban_Texturing/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
