/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Week 01 – Foundations of Physical AI',
      items: [
        'week-01/module-01-01/chapter-01-introduction',
      ],
    },

    {
      type: 'category',
      label: 'Week 02 – Robotic Middleware',
      items: [
        'week-02/module-02-01/chapter-02-ros2',
      ],
    },

    {
      type: 'category',
      label: 'Week 03 – Simulation & Digital Twins',
      items: [
        'week-03/module-03-01/chapter-03-digital-twin',
      ],
    },

    {
      type: 'category',
      label: 'Week 04 – Industrial-Grade Simulation',
      items: [
        'week-04/module-04-01/chapter-04-isaac',
      ],
    },

    {
      type: 'category',
      label: 'Week 05 – Multimodal Intelligence',
      items: [
        'week-05/module-05-01/chapter-05-vla',
      ],
    },

    {
      type: 'category',
      label: 'Week 06 – Integration & Deployment',
      items: [
        'week-06/module-06-01/chapter-06-capstone',
      ],
    },
  ],
};

module.exports = sidebars;
