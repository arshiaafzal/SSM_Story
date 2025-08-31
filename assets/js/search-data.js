// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-",
    title: "",
    section: "Navigation",
    handler: () => {
      window.location.href = "/SSM_Story/";
    },
  },{id: "nav-github",
          title: "github",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/SSM_Story/https:/arshiaafzal.github.io/SSM_Story/";
          },
        },{id: "post-story-of-linear-time-sequence-modeling",
      
        title: "Story of Linear Time Sequence Modeling 📚",
      
      description: "Summary of Linear Transformers",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/SSM_Story/2025/lion-part1-model/";
        
      },
    },{id: "post-lion-part-iv-results",
      
        title: "LION 🦁 Part IV - Results",
      
      description: "Comprehensive results on Vision, MLM and more LION variants",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/SSM_Story/2025/lion-part4-results/";
        
      },
    },{id: "post-lion-part-iii-chunkwise-parallel-from-of-lion",
      
        title: "LION 🦁 Part III - Chunkwise Parallel from of LION",
      
      description: "Explaining LION-Chunk for Balancing Memory-Speed Tradeoffs During Inference",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/SSM_Story/2025/lion-part3-chunk/";
        
      },
    },{id: "post-lion-part-ii-bi-directional-rnn",
      
        title: "LION 🦁 Part II - Bi-directional RNN",
      
      description: "Deriving equivalent bi-directional RNN for Linear Attention",
      section: "Posts",
      handler: () => {
        
          window.location.href = "/SSM_Story/2025/lion-part2-theory/";
        
      },
    },{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/SSM_Story/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/SSM_Story/projects/9_project/";
            },},];
