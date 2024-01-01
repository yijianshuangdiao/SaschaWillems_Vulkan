/*
* Vulkan Example - Basic indexed triangle rendering
*
* Note:
*	This is a "pedal to the metal" example to show off how to get Vulkan up and displaying something
*	Contrary to the other examples, this one won't make use of helper functions or initializers
*	Except in a few cases (swap chain setup e.g.)
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <vector>
#include <exception>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"

// Set to "true" to enable Vulkan's validation layers (see vulkandebug.cpp for details)
#define ENABLE_VALIDATION true
// Set to "true" to use staging buffers for uploading vertex and index data to device local memory
// See "prepareVertices" for details on what's staging and on why to use it
#define USE_STAGING true

class VulkanExample : public VulkanExampleBase
{
public:
	struct Texture
	{
		VkSampler sampler;
		VkImage image;
		VkImageLayout imageLayout;
		VkDeviceMemory deviceMemory;
		VkImageView view;
		uint32_t wight, height;
		uint32_t mipLevels;
	}texture;

	// Vertex layout used in this example
	struct Vertex {
		glm::vec3 position;
		float color[3];
		float uv[2];
	};

	// Vertex buffer and attributes
	struct {
		VkDeviceMemory memory; // Handle to the device memory for this buffer
		VkBuffer buffer;       // Handle to the Vulkan buffer object that the memory is bound to
	} vertices;

	// Index buffer
	struct {
		VkDeviceMemory memory;
		VkBuffer buffer;
		uint32_t count;
	} indices;

	// Uniform buffer block object
	struct {
		VkDeviceMemory memory;
		VkBuffer buffer;
		VkDescriptorBufferInfo descriptor;
	}  uniformBufferVS;

	// For simplicity we use the same uniform block layout as in the shader:
	//
	//	layout(set = 0, binding = 0) uniform UBO
	//	{
	//		mat4 projectionMatrix;
	//		mat4 modelMatrix;
	//		mat4 viewMatrix;
	//	} ubo;
	//
	// This way we can just memcopy the ubo data to the ubo
	// Note: You should use data types that align with the GPU in order to avoid manual padding (vec4, mat4)
	struct {
		glm::mat4 projectionMatrix;
		glm::mat4 modelMatrix;
		glm::mat4 viewMatrix;
	} uboVS;

	//transform
	struct {
		bool enableRotate = true;
		glm::mat4 transform{ 1.0f };
	}cubeTransform;

	// The pipeline layout is used by a pipeline to access the descriptor sets
	// It defines interface (without binding any actual data) between the shader stages used by the pipeline and the shader resources
	// A pipeline layout can be shared among multiple pipelines as long as their interfaces match
	VkPipelineLayout pipelineLayout;

	// Pipelines (often called "pipeline state objects") are used to bake all states that affect a pipeline
	// While in OpenGL every state can be changed at (almost) any time, Vulkan requires to layout the graphics (and compute) pipeline states upfront
	// So for each combination of non-dynamic pipeline states you need a new pipeline (there are a few exceptions to this not discussed here)
	// Even though this adds a new dimension of planning ahead, it's a great opportunity for performance optimizations by the driver
	VkPipeline pipeline;

	// The descriptor set layout describes the shader binding layout (without actually referencing descriptor)
	// Like the pipeline layout it's pretty much a blueprint and can be used with different descriptor sets as long as their layout matches
	VkDescriptorSetLayout descriptorSetLayout;

	// The descriptor set stores the resources bound to the binding points in a shader
	// It connects the binding points of the different shaders with the buffers and images used for those bindings
	VkDescriptorSet descriptorSet;


	// Synchronization primitives
	// Synchronization is an important concept of Vulkan that OpenGL mostly hid away. Getting this right is crucial to using Vulkan.

	// Semaphores
	// Used to coordinate operations within the graphics queue and ensure correct command ordering
	VkSemaphore presentCompleteSemaphore;
	VkSemaphore renderCompleteSemaphore;

	// Fences
	// Used to check the completion of queue operations (e.g. command buffer execution)
	std::vector<VkFence> queueCompleteFences;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - basic lighting";
		// To keep things simple, we don't use the UI overlay
		settings.overlay = true;
		// Setup a default look-at camera
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 12.5f;
		camera.setPosition(glm::vec3(0.0f, 0.0f, -5.5f));
		camera.setRotation(glm::vec3(0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 1.0f, 256.0f);
		// Values not set here are initialized in the base class constructor
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources
		// Note: Inherited destructor cleans up resources stored in base class
		vkDestroyPipeline(device, pipeline, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, vertices.buffer, nullptr);
		vkFreeMemory(device, vertices.memory, nullptr);

		vkDestroyBuffer(device, indices.buffer, nullptr);
		vkFreeMemory(device, indices.memory, nullptr);

		vkDestroyBuffer(device, uniformBufferVS.buffer, nullptr);
		vkFreeMemory(device, uniformBufferVS.memory, nullptr);

		vkDestroySemaphore(device, presentCompleteSemaphore, nullptr);
		vkDestroySemaphore(device, renderCompleteSemaphore, nullptr);

		vkDestroyImageView(device, texture.view, nullptr);
		vkDestroyImage(device, texture.image, nullptr);
		vkDestroySampler(device, texture.sampler, nullptr);
		vkFreeMemory(device, texture.deviceMemory, nullptr);

		for (auto& fence : queueCompleteFences)
		{
			vkDestroyFence(device, fence, nullptr);
		}
	}

	// Enable physical device features required for this example
	virtual void getEnabledFeatures()
	{
		// Enable anisotropic filtering if supported
		if (deviceFeatures.samplerAnisotropy) {
			enabledFeatures.samplerAnisotropy = VK_TRUE;
		};
	}

	// This function is used to request a device memory type that supports all the property flags we request (e.g. device local, host visible)
	// Upon success it will return the index of the memory type that fits our requested memory properties
	// This is necessary as implementations can offer an arbitrary number of memory types with different
	// memory properties.
	// You can check http://vulkan.gpuinfo.org/ for details on different memory configurations
	uint32_t getMemoryTypeIndex(uint32_t typeBits, VkMemoryPropertyFlags properties)
	{
		// Iterate over all memory types available for the device used in this example
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
		{
			if ((typeBits & 1) == 1)
			{
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
				{
					return i;
				}
			}
			typeBits >>= 1;
		}

		throw "Could not find a suitable memory type!";
	}

	// Create the Vulkan synchronization primitives used in this example
	void prepareSynchronizationPrimitives()
	{
		// Semaphores (Used for correct command ordering)
		VkSemaphoreCreateInfo semaphoreCreateInfo = {};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCreateInfo.pNext = nullptr;

		// Semaphore used to ensure that image presentation is complete before starting to submit again
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentCompleteSemaphore));

		// Semaphore used to ensure that all commands submitted have been finished before submitting the image to the queue
		VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderCompleteSemaphore));

		// Fences (Used to check draw command buffer completion)
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		// Create in signaled state so we don't wait on first render of each command buffer
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		queueCompleteFences.resize(drawCmdBuffers.size());
		for (auto& fence : queueCompleteFences)
		{
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		}
	}

	// Get a new command buffer from the command pool
	// If begin is true, the command buffer is also started so we can start adding commands
	VkCommandBuffer getCommandBuffer(bool begin)
	{
		VkCommandBuffer cmdBuffer;

		VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
		cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		cmdBufAllocateInfo.commandPool = cmdPool;
		cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdBufAllocateInfo.commandBufferCount = 1;

		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer));

		// If requested, also start the new command buffer
		if (begin)
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
			VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
		}

		return cmdBuffer;
	}

	// End the command buffer and submit it to the queue
	// Uses a fence to ensure command buffer has finished executing before deleting it
	void flushCommandBuffer(VkCommandBuffer commandBuffer)
	{
		assert(commandBuffer != VK_NULL_HANDLE);

		VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		// Create fence to ensure that the command buffer has finished executing
		VkFenceCreateInfo fenceCreateInfo = {};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = 0;
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		// Wait for the fence to signal that command buffer has finished executing
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
	}

	// Build separate command buffers for every framebuffer image
	// Unlike in OpenGL all rendering commands are recorded once into command buffers that are then resubmitted to the queue
	// This allows to generate work upfront and from multiple threads, one of the biggest advantages of Vulkan
	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = {};
		cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		cmdBufInfo.pNext = nullptr;

		// Set clear values for all framebuffer attachments with loadOp set to clear
		// We use two attachments (color and depth) that are cleared at the start of the subpass and as such we need to set clear values for both
		VkClearValue clearValues[2];
		clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = {};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.pNext = nullptr;
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			// Start the first sub pass specified in our default render pass setup by the base class
			// This will clear the color and depth attachment
			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			// Update dynamic viewport state
			VkViewport viewport = {};
			viewport.height = (float)height;
			viewport.width = (float)width;
			viewport.minDepth = (float) 0.0f;
			viewport.maxDepth = (float) 1.0f;
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			// Update dynamic scissor state
			VkRect2D scissor = {};
			scissor.extent.width = width;
			scissor.extent.height = height;
			scissor.offset.x = 0;
			scissor.offset.y = 0;
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			// Bind descriptor sets describing shader binding points
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

			// Bind the rendering pipeline
			// The pipeline (state object) contains all states of the rendering pipeline, binding it will set all the states specified at pipeline creation time
			vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

			// Bind triangle vertex buffer (contains position and colors)
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &vertices.buffer, offsets);

			// Bind triangle index buffer
			//vkCmdBindIndexBuffer(drawCmdBuffers[i], indices.buffer, 0, VK_INDEX_TYPE_UINT32);

			// Draw indexed triangle
			//vkCmdDrawIndexed(drawCmdBuffers[i], indices.count, 1, 0, 0, 1);
			vkCmdDraw(drawCmdBuffers[i], 36, 36, 0, 0);

			//Draw UI
			drawUI(drawCmdBuffers[i]);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			// Ending the render pass will add an implicit barrier transitioning the frame buffer color attachment to
			// VK_IMAGE_LAYOUT_PRESENT_SRC_KHR for presenting it to the windowing system

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void draw()
	{
#if defined(VK_USE_PLATFORM_MACOS_MVK)
		// SRS - on macOS use swapchain helper function with common semaphores/fences for proper resize handling
		// Get next image in the swap chain (back/front buffer)
		prepareFrame();

		// Use a fence to wait until the command buffer has finished execution before using it again
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &waitFences[currentBuffer], VK_TRUE, UINT64_MAX));
		VK_CHECK_RESULT(vkResetFences(device, 1, &waitFences[currentBuffer]));
#else
		// SRS - on other platforms use original bare code with local semaphores/fences for illustrative purposes
		// Get next image in the swap chain (back/front buffer)
		VkResult acquire = swapChain.acquireNextImage(presentCompleteSemaphore, &currentBuffer);
		if (!((acquire == VK_SUCCESS) || (acquire == VK_SUBOPTIMAL_KHR))) {
			VK_CHECK_RESULT(acquire);
		}

		// Use a fence to wait until the command buffer has finished execution before using it again
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &queueCompleteFences[currentBuffer], VK_TRUE, UINT64_MAX));
		VK_CHECK_RESULT(vkResetFences(device, 1, &queueCompleteFences[currentBuffer]));
#endif

		// Pipeline stage at which the queue submission will wait (via pWaitSemaphores)
		VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		// The submit info structure specifies a command buffer queue submission batch
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pWaitDstStageMask = &waitStageMask;               // Pointer to the list of pipeline stages that the semaphore waits will occur at
		submitInfo.waitSemaphoreCount = 1;                           // One wait semaphore
		submitInfo.signalSemaphoreCount = 1;                         // One signal semaphore
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer]; // Command buffers(s) to execute in this batch (submission)
		submitInfo.commandBufferCount = 1;                           // One command buffer

#if defined(VK_USE_PLATFORM_MACOS_MVK)
		// SRS - on macOS use swapchain helper function with common semaphores/fences for proper resize handling
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;    // Semaphore(s) to wait upon before the submitted command buffer starts executing
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;   // Semaphore(s) to be signaled when command buffers have completed

		// Submit to the graphics queue passing a wait fence
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, waitFences[currentBuffer]));

		// Present the current buffer to the swap chain
		submitFrame();
#else
		// SRS - on other platforms use original bare code with local semaphores/fences for illustrative purposes
		submitInfo.pWaitSemaphores = &presentCompleteSemaphore;      // Semaphore(s) to wait upon before the submitted command buffer starts executing
		submitInfo.pSignalSemaphores = &renderCompleteSemaphore;     // Semaphore(s) to be signaled when command buffers have completed

		// Submit to the graphics queue passing a wait fence
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, queueCompleteFences[currentBuffer]));

		// Present the current buffer to the swap chain
		// Pass the semaphore signaled by the command buffer submission from the submit info as the wait semaphore for swap chain presentation
		// This ensures that the image is not presented to the windowing system until all commands have been submitted
		VkResult present = swapChain.queuePresent(queue, currentBuffer, renderCompleteSemaphore);
		if (!((present == VK_SUCCESS) || (present == VK_SUBOPTIMAL_KHR))) {
			VK_CHECK_RESULT(present);
		}
		VK_CHECK_RESULT(vkQueueWaitIdle(queue));
#endif
	}

	void loadTexture()
	{
		std::string filename = getAssetPath() + "textures/metalplate01_rgba.ktx";
		VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

		ktxResult result;
		ktxTexture* ktxTexture;

#if defined(__ANDROID__)
		// Textures are stored inside the apk on Android (compressed)
		// So they need to be loaded via the asset manager
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		if (!asset) {
			vks::tools::exitFatal("Could not load texture from " + filename + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		}
		size_t size = AAsset_getLength(asset);
		assert(size > 0);

		ktx_uint8_t* textureData = new ktx_uint8_t[size];
		AAsset_read(asset, textureData, size);
		AAsset_close(asset);
		result = ktxTexture_CreateFromMemory(textureData, size, KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
		delete[] textureData;
#else
		if (!vks::tools::fileExists(filename)) {
			vks::tools::exitFatal("Could not load texture from " + filename + "\n\nMake sure the assets submodule has been checked out and is up-to-date.", -1);
		}
		result = ktxTexture_CreateFromNamedFile(filename.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTexture);
#endif
		assert(result == KTX_SUCCESS);

		//从ktx纹理对象中获取使用和上传纹理数据所需的属性
		texture.wight = ktxTexture->baseWidth;
		texture.height = ktxTexture->baseHeight;
		texture.mipLevels = ktxTexture->numLevels;
		ktx_uint8_t* ktxTextureData = ktxTexture_GetData(ktxTexture);
		ktx_size_t ktxTextureSize =  ktxTexture_GetSize(ktxTexture);

		VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
		VkMemoryRequirements memReqs = {};

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;

		VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo();
		bufferCreateInfo.size = ktxTextureSize;
		bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &stagingBuffer));

		vkGetBufferMemoryRequirements(device, stagingBuffer, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &stagingMemory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0));

		uint8_t *data;
		VK_CHECK_RESULT(vkMapMemory(device, stagingMemory, 0, memReqs.size, 0, (void**)&data));
		memcpy(data, ktxTextureData, ktxTextureSize);
		vkUnmapMemory(device, stagingMemory);

		std::vector<VkBufferImageCopy>bufferCopyRegions;
		//uint32_t offset = 0;

		for (uint32_t i=0;i<texture.mipLevels;i++)
		{
			ktx_size_t offset;
			KTX_error_code ret = ktxTexture_GetImageOffset(ktxTexture, i, 0, 0, &offset);
			assert(ret == KTX_SUCCESS);

			VkBufferImageCopy bufferCopyRegion = {};
			bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
			bufferCopyRegion.imageSubresource.mipLevel = i;
			bufferCopyRegion.imageSubresource.layerCount = 1;
			bufferCopyRegion.imageExtent.width = ktxTexture->baseWidth >> i;
			bufferCopyRegion.imageExtent.height = ktxTexture->baseHeight >> i;
			bufferCopyRegion.imageExtent.depth = 1;
			bufferCopyRegion.bufferOffset = offset;
			bufferCopyRegions.push_back(bufferCopyRegion);
		}

		VkImageCreateInfo imageCreateInfo = vks::initializers::imageCreateInfo();
		imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
		imageCreateInfo.format = format;
		imageCreateInfo.mipLevels = texture.mipLevels;
		imageCreateInfo.arrayLayers = 1;
		imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageCreateInfo.extent = { texture.wight,texture.height,1 };
		imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		VK_CHECK_RESULT(vkCreateImage(device, &imageCreateInfo, nullptr, &texture.image));

		vkGetImageMemoryRequirements(device, texture.image, &memReqs);
		memAllocInfo.allocationSize = memReqs.size;
		memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAllocInfo, nullptr, &texture.deviceMemory));
		VK_CHECK_RESULT(vkBindImageMemory(device, texture.image, texture.deviceMemory, 0));

		VkCommandBuffer copyCmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

		VkImageSubresourceRange subresourceRange = {};
		// Image only contains color data
		subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		// Start at first mip level
		subresourceRange.baseMipLevel = 0;
		// We will transition on all mip levels
		subresourceRange.levelCount = texture.mipLevels;
		// The 2D texture only has one layer
		subresourceRange.layerCount = 1;

		VkImageMemoryBarrier imageMemoryBarrier = vks::initializers::imageMemoryBarrier();;
		imageMemoryBarrier.image = texture.image;
		imageMemoryBarrier.subresourceRange = subresourceRange;
		imageMemoryBarrier.srcAccessMask = 0;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

		vkCmdPipelineBarrier(
			copyCmd,
			VK_PIPELINE_STAGE_HOST_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);

		vkCmdCopyBufferToImage(
			copyCmd,
			stagingBuffer,
			texture.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			static_cast<uint32_t>(bufferCopyRegions.size()),
			bufferCopyRegions.data());

		imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		vkCmdPipelineBarrier(
			copyCmd,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &imageMemoryBarrier);

		texture.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		vkFreeMemory(device, stagingMemory, nullptr);
		vkDestroyBuffer(device, stagingBuffer, nullptr);

		ktxTexture_Destroy(ktxTexture);

		// Create a texture sampler
		// In Vulkan textures are accessed by samplers
		// This separates all the sampling information from the texture data. This means you could have multiple sampler objects for the same texture with different settings
		// Note: Similar to the samplers available with OpenGL 3.3
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = VK_FILTER_LINEAR;
		sampler.minFilter = VK_FILTER_LINEAR;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		sampler.mipLodBias = 0.0f;
		sampler.compareOp = VK_COMPARE_OP_NEVER;
		sampler.minLod = 0.0f;
		// Set max level-of-detail to mip level count of the texture
		sampler.maxLod = (float)texture.mipLevels ;
		// Enable anisotropic filtering
		// This feature is optional, so we must check if it's supported on the device
		if (vulkanDevice->features.samplerAnisotropy) {
			// Use max. level of anisotropy for this example
			sampler.maxAnisotropy = vulkanDevice->properties.limits.maxSamplerAnisotropy;
			sampler.anisotropyEnable = VK_TRUE;
		}
		else {
			// The device does not support anisotropic filtering
			sampler.maxAnisotropy = 1.0;
			sampler.anisotropyEnable = VK_FALSE;
		}
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &texture.sampler));

		// Create image view
		// Textures are not directly accessed by the shaders and
		// are abstracted by image views containing additional
		// information and sub resource ranges
		VkImageViewCreateInfo view = vks::initializers::imageViewCreateInfo();
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = format;
		// The subresource range describes the set of mip levels (and array layers) that can be accessed through this image view
		// It's possible to create multiple image views for a single image referring to different (and/or overlapping) ranges of the image
		view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		view.subresourceRange.baseMipLevel = 0;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.layerCount = 1;
		// Linear tiling usually won't support mip maps
		// Only set mip map count if optimal tiling is used
		view.subresourceRange.levelCount = texture.mipLevels;
		// The view will be based on the texture's image
		view.image = texture.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &view, nullptr, &texture.view));
	}

	// Prepare vertex and index buffers for an indexed triangle
	// Also uploads them to device local memory using staging and initializes vertex input and attribute binding to match the vertex shader
	void prepareVertices(bool useStagingBuffers)
	{
		// A note on memory management in Vulkan in general:
		//	This is a very complex topic and while it's fine for an example application to small individual memory allocations that is not
		//	what should be done a real-world application, where you should allocate large chunks of memory at once instead.

		//逆时针Uv
		std::vector <glm::vec2> quadUv = { {0.0f,0.0f},{1.0f,0.0f},{1.0f,1.0f},{1.0f,1.0f},{0.0f,1.0f},{0.0f,0.0f} };

		// Setup vertices
		std::vector<Vertex> vertexBuffer =
		{
			{ {  -1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f },{}},
			{ { 1.0f, -1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f },{} },
			{ { 1.0f, -1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f },{}},
			{{ -1.0f, -1.0f, -1.0f}, { 1.0f, 0.0f, 1.0f },{}},
			{ {  -1.0f,  1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f },{}},
			{ {  1.0f, 1.0f, 1.0f}, { 1.0f, 0.0f, 1.0f },{} },
			{ { 1.0f, 1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f },{}},
			{ { -1.0f, 1.0f, -1.0f }, { 1.0f, 0.0f, 1.0f },{}}
		};

		// Setup indices
		std::vector<uint32_t> indexBuffer = {
			2,1,0,0,3,2,//下
			4,5,6,6,7,4,//上
			7,3,0,0,4,7,//左
			1,2,6,6,5,1,//右
			0,1,5,5,4,0,//前
			2,3,7,7,6,2 //后
		};

		//quad vertices
		std::vector<Vertex> quadVertexBuffer = {};
		for (int i = 0; i < indexBuffer.size(); ++i)
		{
			const auto pos = vertexBuffer[indexBuffer[i]].position;
			const auto uv = quadUv[i % 6];
			quadVertexBuffer.push_back({ {pos.x,pos.y,pos.z},{},{uv.x, uv.y} });
		}
		vertexBuffer.resize(indexBuffer.size());
		std::swap(vertexBuffer, quadVertexBuffer);

		uint32_t vertexBufferSize = static_cast<uint32_t>(vertexBuffer.size()) * sizeof(Vertex);
		
		indices.count = static_cast<uint32_t>(indexBuffer.size());
		uint32_t indexBufferSize = indices.count * sizeof(uint32_t);

		VkMemoryAllocateInfo memAlloc = {};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		VkMemoryRequirements memReqs;

		void *data;

		// Static data like vertex and index buffer should be stored on the device memory
		// for optimal (and fastest) access by the GPU
		//
		// To achieve this we use so-called "staging buffers" :
		// - Create a buffer that's visible to the host (and can be mapped)
		// - Copy the data to this buffer
		// - Create another buffer that's local on the device (VRAM) with the same size
		// - Copy the data from the host to the device using a command buffer
		// - Delete the host visible (staging) buffer
		// - Use the device local buffers for rendering

		struct StagingBuffer {
			VkDeviceMemory memory;
			VkBuffer buffer;
		};

		struct {
			StagingBuffer vertices;
			StagingBuffer indices;
		} stagingBuffers;

		// Vertex buffer
		VkBufferCreateInfo vertexBufferInfo = {};
		vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		vertexBufferInfo.size = vertexBufferSize;
		// Buffer is used as the copy source
		vertexBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Create a host-visible buffer to copy the vertex data to (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfo, nullptr, &stagingBuffers.vertices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		// Request a host visible memory type that can be used to copy our data do
		// Also request it to be coherent, so that writes are visible to the GPU right after unmapping the buffer
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.vertices.memory));
		// Map and copy
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.vertices.memory, 0, memAlloc.allocationSize, 0, &data));
		memcpy(data, vertexBuffer.data(), vertexBufferSize);
		vkUnmapMemory(device, stagingBuffers.vertices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.vertices.buffer, stagingBuffers.vertices.memory, 0));

		// Create a device local buffer to which the (host local) vertex data will be copied and which will be used for rendering
		vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &vertexBufferInfo, nullptr, &vertices.buffer));
		vkGetBufferMemoryRequirements(device, vertices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &vertices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, vertices.buffer, vertices.memory, 0));

		// Index buffer
		VkBufferCreateInfo indexbufferInfo = {};
		indexbufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		indexbufferInfo.size = indexBufferSize;
		indexbufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		// Copy index data to a buffer visible to the host (staging buffer)
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferInfo, nullptr, &stagingBuffers.indices.buffer));
		vkGetBufferMemoryRequirements(device, stagingBuffers.indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &stagingBuffers.indices.memory));
		VK_CHECK_RESULT(vkMapMemory(device, stagingBuffers.indices.memory, 0, indexBufferSize, 0, &data));
		memcpy(data, indexBuffer.data(), indexBufferSize);
		vkUnmapMemory(device, stagingBuffers.indices.memory);
		VK_CHECK_RESULT(vkBindBufferMemory(device, stagingBuffers.indices.buffer, stagingBuffers.indices.memory, 0));

		// Create destination buffer with device only visibility
		indexbufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		VK_CHECK_RESULT(vkCreateBuffer(device, &indexbufferInfo, nullptr, &indices.buffer));
		vkGetBufferMemoryRequirements(device, indices.buffer, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &indices.memory));
		VK_CHECK_RESULT(vkBindBufferMemory(device, indices.buffer, indices.memory, 0));

		// Buffer copies have to be submitted to a queue, so we need a command buffer for them
		// Note: Some devices offer a dedicated transfer queue (with only the transfer bit set) that may be faster when doing lots of copies
		VkCommandBuffer copyCmd = getCommandBuffer(true);

		// Put buffer region copies into command buffer
		VkBufferCopy copyRegion = {};

		// Vertex buffer
		copyRegion.size = vertexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.vertices.buffer, vertices.buffer, 1, &copyRegion);
		// Index buffer
		copyRegion.size = indexBufferSize;
		vkCmdCopyBuffer(copyCmd, stagingBuffers.indices.buffer, indices.buffer,	1, &copyRegion);

		// Flushing the command buffer will also submit it to the queue and uses a fence to ensure that all commands have been executed before returning
		flushCommandBuffer(copyCmd);

		// Destroy staging buffers
		// Note: Staging buffer must not be deleted before the copies have been submitted and executed
		vkDestroyBuffer(device, stagingBuffers.vertices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.vertices.memory, nullptr);
		vkDestroyBuffer(device, stagingBuffers.indices.buffer, nullptr);
		vkFreeMemory(device, stagingBuffers.indices.memory, nullptr);
	}

	void setupDescriptorPool()
	{
		// We need to tell the API the number of max. requested descriptors per type
		VkDescriptorPoolSize typeCounts[2];
		// This example only uses one descriptor type (uniform buffer) and only requests one descriptor of this type
		typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		typeCounts[0].descriptorCount = 1;
		// For additional types you need to add new entries in the type count list
		// E.g. for two combined image samplers :
		typeCounts[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		typeCounts[1].descriptorCount = 1;

		// Create the global descriptor pool
		// All descriptors used in this example are allocated from this pool
		VkDescriptorPoolCreateInfo descriptorPoolInfo = {};
		descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolInfo.pNext = nullptr;
		descriptorPoolInfo.poolSizeCount = 2;
		descriptorPoolInfo.pPoolSizes = typeCounts;
		// Set the max. number of descriptor sets that can be requested from this pool (requesting beyond this limit will result in an error)
		descriptorPoolInfo.maxSets = 2;

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		// Setup layout of descriptors used in this example
		// Basically connects the different shader stages to descriptors for binding uniform buffers, image samplers, etc.
		// So every shader binding should map to one descriptor set layout binding

		std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {
			// Binding 0: Uniform buffer (Vertex shader)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				VK_SHADER_STAGE_VERTEX_BIT,
				0),
			// Binding 1: Uniform buffer (Fragment shader)
			vks::initializers::descriptorSetLayoutBinding(
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				1)
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout = {};
		descriptorLayout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorLayout.pNext = nullptr;
		descriptorLayout.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		descriptorLayout.pBindings = layoutBindings.data();

		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		// Create the pipeline layout that is used to generate the rendering pipelines that are based on this descriptor set layout
		// In a more complex scenario you would have different pipeline layouts for different descriptor set layouts that could be reused
		VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = {};
		pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pPipelineLayoutCreateInfo.pNext = nullptr;
		pPipelineLayoutCreateInfo.setLayoutCount = 1;
		pPipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));
	}

	void setupDescriptorSet()
	{
		// Allocate a new descriptor set from the global descriptor pool
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptorSetLayout;

		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

		// Setup a descriptor image info for the current texture to be used as a combined image sampler
		VkDescriptorImageInfo textureDescriptor;
		textureDescriptor.imageView = texture.view;				// The image's view (images are never directly accessed by the shader, but rather through views defining subresources)
		textureDescriptor.sampler = texture.sampler;			// The sampler (Telling the pipeline how to sample the texture, including repeat, border, etc.)
		textureDescriptor.imageLayout = texture.imageLayout;	// The current layout of the image (Note: Should always fit the actual use, e.g. shader read)

		// Update the descriptor set determining the shader binding points
		// For every binding point used in a shader there needs to be one
		// descriptor set matching that binding point

		std::vector<VkWriteDescriptorSet>writeDescriptorSets = {
			// Binding 0 : Uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				0,
				&uniformBufferVS.descriptor
			),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSet,
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				1,
				&textureDescriptor
			),
		};

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
	}

	// Create the depth (and stencil) buffer attachments used by our framebuffers
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupDepthStencil()
	{
		// Create an optimal image used as the depth stencil attachment
		VkImageCreateInfo image = {};
		image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		image.imageType = VK_IMAGE_TYPE_2D;
		image.format = depthFormat;
		// Use example's height and width
		image.extent = { width, height, 1 };
		image.mipLevels = 1;
		image.arrayLayers = 1;
		image.samples = VK_SAMPLE_COUNT_1_BIT;
		image.tiling = VK_IMAGE_TILING_OPTIMAL;
		image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &depthStencil.image));

		// Allocate memory for the image (device local) and bind it to our image
		VkMemoryAllocateInfo memAlloc = {};
		memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		VkMemoryRequirements memReqs;
		vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &depthStencil.mem));
		VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

		// Create a view for the depth stencil image
		// Images aren't directly accessed in Vulkan, but rather through views described by a subresource range
		// This allows for multiple views of one image with differing ranges (e.g. for different layers)
		VkImageViewCreateInfo depthStencilView = {};
		depthStencilView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
		depthStencilView.format = depthFormat;
		depthStencilView.subresourceRange = {};
		depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		// Stencil aspect should only be set on depth + stencil formats (VK_FORMAT_D16_UNORM_S8_UINT..VK_FORMAT_D32_SFLOAT_S8_UINT)
		if (depthFormat >= VK_FORMAT_D16_UNORM_S8_UINT)
			depthStencilView.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;

		depthStencilView.subresourceRange.baseMipLevel = 0;
		depthStencilView.subresourceRange.levelCount = 1;
		depthStencilView.subresourceRange.baseArrayLayer = 0;
		depthStencilView.subresourceRange.layerCount = 1;
		depthStencilView.image = depthStencil.image;
		VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &depthStencil.view));
	}

	// Create a frame buffer for each swap chain image
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupFrameBuffer()
	{
		// Create a frame buffer for every image in the swapchain
		frameBuffers.resize(swapChain.imageCount);
		for (size_t i = 0; i < frameBuffers.size(); i++)
		{
			std::array<VkImageView, 2> attachments;
			attachments[0] = swapChain.buffers[i].view; // Color attachment is the view of the swapchain image
			attachments[1] = depthStencil.view;         // Depth/Stencil attachment is the same for all frame buffers

			VkFramebufferCreateInfo frameBufferCreateInfo = {};
			frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			// All frame buffers use the same renderpass setup
			frameBufferCreateInfo.renderPass = renderPass;
			frameBufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			frameBufferCreateInfo.pAttachments = attachments.data();
			frameBufferCreateInfo.width = width;
			frameBufferCreateInfo.height = height;
			frameBufferCreateInfo.layers = 1;
			// Create the framebuffer
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
		}
	}

	// Render pass setup
	// Render passes are a new concept in Vulkan. They describe the attachments used during rendering and may contain multiple subpasses with attachment dependencies
	// This allows the driver to know up-front what the rendering will look like and is a good opportunity to optimize especially on tile-based renderers (with multiple subpasses)
	// Using sub pass dependencies also adds implicit layout transitions for the attachment used, so we don't need to add explicit image memory barriers to transform them
	// Note: Override of virtual function in the base class and called from within VulkanExampleBase::prepare
	void setupRenderPass()
	{
		// This example will use a single render pass with one subpass

		// Descriptors for the attachments used by this renderpass
		std::array<VkAttachmentDescription, 2> attachments = {};

		// Color attachment
		attachments[0].format = swapChain.colorFormat;                                  // Use the color format selected by the swapchain
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;                                 // We don't use multi sampling in this example
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                            // Clear this attachment at the start of the render pass
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;                          // Keep its contents after the render pass is finished (for displaying it)
		attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                 // We don't use stencil, so don't care for load
		attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;               // Same for store
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                       // Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;                   // Layout to which the attachment is transitioned when the render pass is finished
		                                                                                // As we want to present the color buffer to the swapchain, we transition to PRESENT_KHR
		// Depth attachment
		attachments[1].format = depthFormat;                                           // A proper depth format is selected in the example base
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                           // Clear depth at start of first subpass
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;                     // We don't need depth after render pass has finished (DONT_CARE may result in better performance)
		attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;                // No stencil
		attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;              // No Stencil
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;                      // Layout at render pass start. Initial doesn't matter, so we use undefined
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Transition to depth/stencil attachment

		// Setup attachment references
		VkAttachmentReference colorReference = {};
		colorReference.attachment = 0;                                    // Attachment 0 is color
		colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Attachment layout used as color during the subpass

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 1;                                            // Attachment 1 is color
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Attachment used as depth/stencil used during the subpass

		// Setup a single subpass reference
		VkSubpassDescription subpassDescription = {};
		subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpassDescription.colorAttachmentCount = 1;                            // Subpass uses one color attachment
		subpassDescription.pColorAttachments = &colorReference;                 // Reference to the color attachment in slot 0
		subpassDescription.pDepthStencilAttachment = &depthReference;           // Reference to the depth attachment in slot 1
		subpassDescription.inputAttachmentCount = 0;                            // Input attachments can be used to sample from contents of a previous subpass
		subpassDescription.pInputAttachments = nullptr;                         // (Input attachments not used by this example)
		subpassDescription.preserveAttachmentCount = 0;                         // Preserved attachments can be used to loop (and preserve) attachments through subpasses
		subpassDescription.pPreserveAttachments = nullptr;                      // (Preserve attachments not used by this example)
		subpassDescription.pResolveAttachments = nullptr;                       // Resolve attachments are resolved at the end of a sub pass and can be used for e.g. multi sampling

		// Setup subpass dependencies
		// These will add the implicit attachment layout transitions specified by the attachment descriptions
		// The actual usage layout is preserved through the layout specified in the attachment reference
		// Each subpass dependency will introduce a memory and execution dependency between the source and dest subpass described by
		// srcStageMask, dstStageMask, srcAccessMask, dstAccessMask (and dependencyFlags is set)
		// Note: VK_SUBPASS_EXTERNAL is a special constant that refers to all commands executed outside of the actual renderpass)
		std::array<VkSubpassDependency, 2> dependencies;

		// Does the transition from final to initial layout for the depth an color attachments
		// Depth attachment
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
		dependencies[0].dependencyFlags = 0;
		// Color attachment
		dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].dstSubpass = 0;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].srcAccessMask = 0;
		dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
		dependencies[1].dependencyFlags = 0;

		// Create the actual renderpass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());  // Number of attachments used by this render pass
		renderPassInfo.pAttachments = attachments.data();                            // Descriptions of the attachments used by the render pass
		renderPassInfo.subpassCount = 1;                                             // We only use one subpass in this example
		renderPassInfo.pSubpasses = &subpassDescription;                             // Description of that subpass
		renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size()); // Number of subpass dependencies
		renderPassInfo.pDependencies = dependencies.data();                          // Subpass dependencies used by the render pass

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
	}

	// Vulkan loads its shaders from an immediate binary representation called SPIR-V
	// Shaders are compiled offline from e.g. GLSL using the reference glslang compiler
	// This function loads such a shader from a binary file and returns a shader module structure
	VkShaderModule loadSPIRVShader(std::string filename)
	{
		size_t shaderSize;
		char* shaderCode = NULL;

#if defined(__ANDROID__)
		// Load shader from compressed asset
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		shaderSize = AAsset_getLength(asset);
		assert(shaderSize > 0);

		shaderCode = new char[shaderSize];
		AAsset_read(asset, shaderCode, shaderSize);
		AAsset_close(asset);
#else
		std::ifstream is(filename, std::ios::binary | std::ios::in | std::ios::ate);

		if (is.is_open())
		{
			shaderSize = is.tellg();
			is.seekg(0, std::ios::beg);
			// Copy file contents into a buffer
			shaderCode = new char[shaderSize];
			is.read(shaderCode, shaderSize);
			is.close();
			assert(shaderSize > 0);
		}
#endif
		if (shaderCode)
		{
			// Create a new shader module that will be used for pipeline creation
			VkShaderModuleCreateInfo moduleCreateInfo{};
			moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			moduleCreateInfo.codeSize = shaderSize;
			moduleCreateInfo.pCode = (uint32_t*)shaderCode;

			VkShaderModule shaderModule;
			VK_CHECK_RESULT(vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule));

			delete[] shaderCode;

			return shaderModule;
		}
		else
		{
			std::cerr << "Error: Could not open shader file \"" << filename << "\"" << std::endl;
			return VK_NULL_HANDLE;
		}
	}

	void preparePipelines()
	{
		// Create the graphics pipeline used in this example
		// Vulkan uses the concept of rendering pipelines to encapsulate fixed states, replacing OpenGL's complex state machine
		// A pipeline is then stored and hashed on the GPU making pipeline changes very fast
		// Note: There are still a few dynamic states that are not directly part of the pipeline (but the info that they are used is)

		VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
		pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		// The layout used for this pipeline (can be shared among multiple pipelines using the same layout)
		pipelineCreateInfo.layout = pipelineLayout;
		// Renderpass this pipeline is attached to
		pipelineCreateInfo.renderPass = renderPass;

		// Construct the different states making up the pipeline

		// Input assembly state describes how primitives are assembled
		// This pipeline will assemble vertex data as a triangle lists (though we only use one triangle)
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
		inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterizationState = {};
		rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizationState.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizationState.depthClampEnable = VK_FALSE;
		rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		rasterizationState.depthBiasEnable = VK_FALSE;
		rasterizationState.lineWidth = 1.0f;

		// Color blend state describes how blend factors are calculated (if used)
		// We need one blend attachment state per color attachment (even if blending is not used)
		VkPipelineColorBlendAttachmentState blendAttachmentState[1] = {};
		blendAttachmentState[0].colorWriteMask = 0xf;
		blendAttachmentState[0].blendEnable = VK_FALSE;
		VkPipelineColorBlendStateCreateInfo colorBlendState = {};
		colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlendState.attachmentCount = 1;
		colorBlendState.pAttachments = blendAttachmentState;

		// Viewport state sets the number of viewports and scissor used in this pipeline
		// Note: This is actually overridden by the dynamic states (see below)
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		// Enable dynamic states
		// Most states are baked into the pipeline, but there are still a few dynamic states that can be changed within a command buffer
		// To be able to change these we need do specify which dynamic states will be changed using this pipeline. Their actual states are set later on in the command buffer.
		// For this example we will set the viewport and scissor using dynamic states
		std::vector<VkDynamicState> dynamicStateEnables;
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
		dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
		VkPipelineDynamicStateCreateInfo dynamicState = {};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.pDynamicStates = dynamicStateEnables.data();
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());

		// Depth and stencil state containing depth and stencil compare and test operations
		// We only use depth tests and want depth tests and writes to be enabled and compare with less or equal
		VkPipelineDepthStencilStateCreateInfo depthStencilState = {};
		depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencilState.depthTestEnable = VK_TRUE;
		depthStencilState.depthWriteEnable = VK_TRUE;
		depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthStencilState.depthBoundsTestEnable = VK_FALSE;
		depthStencilState.back.failOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.passOp = VK_STENCIL_OP_KEEP;
		depthStencilState.back.compareOp = VK_COMPARE_OP_ALWAYS;
		depthStencilState.stencilTestEnable = VK_FALSE;
		depthStencilState.front = depthStencilState.back;

		// Multi sampling state
		// This example does not make use of multi sampling (for anti-aliasing), the state must still be set and passed to the pipeline
		VkPipelineMultisampleStateCreateInfo multisampleState = {};
		multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampleState.pSampleMask = nullptr;

		// Vertex input descriptions
		// Specifies the vertex input parameters for a pipeline

		// Vertex input binding
		// This example uses a single vertex input binding at binding point 0 (see vkCmdBindVertexBuffers)
		VkVertexInputBindingDescription vertexInputBinding = {};
		vertexInputBinding.binding = 0;
		vertexInputBinding.stride = sizeof(Vertex);
		vertexInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Input attribute bindings describe shader attribute locations and memory layouts
		std::array<VkVertexInputAttributeDescription, 3> vertexInputAttributs;
		// These match the following shader layout (see triangle.vert):
		//	layout (location = 0) in vec3 inPos;
		//	layout (location = 1) in vec3 inColor;
		// Attribute location 0: Position
		vertexInputAttributs[0].binding = 0;
		vertexInputAttributs[0].location = 0;
		// Position attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[0].offset = offsetof(Vertex, position);
		// Attribute location 1: Color
		vertexInputAttributs[1].binding = 0;
		vertexInputAttributs[1].location = 1;
		// Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32 B32)
		vertexInputAttributs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexInputAttributs[1].offset = offsetof(Vertex, color);
		// Attribute location 2: UV
		vertexInputAttributs[2].binding = 0;
		vertexInputAttributs[2].location = 2;
		// Color attribute is three 32 bit signed (SFLOAT) floats (R32 G32)
		vertexInputAttributs[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexInputAttributs[2].offset = offsetof(Vertex, uv);

		// Vertex input state used for pipeline creation
		VkPipelineVertexInputStateCreateInfo vertexInputState = {};
		vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputState.vertexBindingDescriptionCount = 1;
		vertexInputState.pVertexBindingDescriptions = &vertexInputBinding;
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributs.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributs.data();

		// Shaders
		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages{};

		// Vertex shader
		shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		// Load binary SPIR-V shader
		shaderStages[0].module = loadSPIRVShader(getShadersPath() + "basiclighting/basiclighting.vert.spv");
		// Main entry point for the shader
		shaderStages[0].pName = "main";
		assert(shaderStages[0].module != VK_NULL_HANDLE);

		// Fragment shader
		shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// Set pipeline stage for this shader
		shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		// Load binary SPIR-V shader
		shaderStages[1].module = loadSPIRVShader(getShadersPath() + "basiclighting/basiclighting.frag.spv");
		// Main entry point for the shader
		shaderStages[1].pName = "main";
		assert(shaderStages[1].module != VK_NULL_HANDLE);

		// Set pipeline shader stage info
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		// Assign the pipeline states to the pipeline creation info structure
		pipelineCreateInfo.pVertexInputState = &vertexInputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;

		// Create rendering pipeline using the specified states
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));

		// Shader modules are no longer needed once the graphics pipeline has been created
		vkDestroyShaderModule(device, shaderStages[0].module, nullptr);
		vkDestroyShaderModule(device, shaderStages[1].module, nullptr);
	}

	void prepareUniformBuffers()
	{
		// Prepare and initialize a uniform buffer block containing shader uniforms
		// Single uniforms like in OpenGL are no longer present in Vulkan. All Shader uniforms are passed via uniform buffer blocks
		VkMemoryRequirements memReqs;

		// Vertex shader uniform buffer block
		VkBufferCreateInfo bufferInfo = {};
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.pNext = nullptr;
		allocInfo.allocationSize = 0;
		allocInfo.memoryTypeIndex = 0;

		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = sizeof(uboVS);
		// This buffer will be used as a uniform buffer
		bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

		// Create a new buffer
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferInfo, nullptr, &uniformBufferVS.buffer));
		// Get memory requirements including size, alignment and memory type
		vkGetBufferMemoryRequirements(device, uniformBufferVS.buffer, &memReqs);
		allocInfo.allocationSize = memReqs.size;
		// Get the memory type index that supports host visible memory access
		// Most implementations offer multiple memory types and selecting the correct one to allocate memory from is crucial
		// We also want the buffer to be host coherent so we don't have to flush (or sync after every update.
		// Note: This may affect performance so you might not want to do this in a real world application that updates buffers on a regular base
		allocInfo.memoryTypeIndex = getMemoryTypeIndex(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		// Allocate memory for the uniform buffer
		VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &(uniformBufferVS.memory)));
		// Bind memory to buffer
		VK_CHECK_RESULT(vkBindBufferMemory(device, uniformBufferVS.buffer, uniformBufferVS.memory, 0));

		// Store information in the uniform's descriptor that is used by the descriptor set
		uniformBufferVS.descriptor.buffer = uniformBufferVS.buffer;
		uniformBufferVS.descriptor.offset = 0;
		uniformBufferVS.descriptor.range = sizeof(uboVS);

		updateUniformBuffers();
	}

	void updateUniformBuffers()
	{
		// Pass matrices to the shaders
		uboVS.projectionMatrix = camera.matrices.perspective;
		uboVS.viewMatrix = camera.matrices.view;
		//uboVS.modelMatrix = glm::mat4(1.0f);

		//rotate
		const float angle = cubeTransform.enableRotate ? 0.01f : 0.0f;
		cubeTransform.transform = glm::rotate(cubeTransform.transform, glm::radians(angle), glm::vec3(1.0, -1.0, 1.0));
		uboVS.modelMatrix = cubeTransform.transform;

		// Map uniform buffer and update it
		uint8_t *pData;
		VK_CHECK_RESULT(vkMapMemory(device, uniformBufferVS.memory, 0, sizeof(uboVS), 0, (void **)&pData));
		memcpy(pData, &uboVS, sizeof(uboVS));
		// Unmap after data has been copied
		// Note: Since we requested a host coherent memory type for the uniform buffer, the write is instantly visible to the GPU
		vkUnmapMemory(device, uniformBufferVS.memory);
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		prepareSynchronizationPrimitives();
		loadTexture();
		prepareVertices(USE_STAGING);
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused) {
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		// This function is called by the base example class each time the view is changed by user input
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay* overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->checkBox("Rotate", &cubeTransform.enableRotate)) {
				//
			}
		}
	}
};

// OS specific macros for the example main entry points
// Most of the code base is shared for the different supported operating systems, but stuff like message handling differs

#if defined(_WIN32)
// Windows entry point
VulkanExample *vulkanExample;
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleMessages(hWnd, uMsg, wParam, lParam);
	}
	return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR pCmdLine, int nCmdShow)
{
	for (size_t i = 0; i < __argc; i++) { VulkanExample::args.push_back(__argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow(hInstance, WndProc);
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}

#elif defined(__ANDROID__)
// Android entry point
VulkanExample *vulkanExample;
void android_main(android_app* state)
{
	vulkanExample = new VulkanExample();
	state->userData = vulkanExample;
	state->onAppCmd = VulkanExample::handleAppCommand;
	state->onInputEvent = VulkanExample::handleAppInput;
	androidApp = state;
	vulkanExample->renderLoop();
	delete(vulkanExample);
}
#elif defined(_DIRECT2DISPLAY)

// Linux entry point with direct to display wsi
// Direct to Displays (D2D) is used on embedded platforms
VulkanExample *vulkanExample;
static void handleEvent()
{
}
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
VulkanExample *vulkanExample;
static void handleEvent(const DFBWindowEvent *event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif defined(__linux__) || defined(__FreeBSD__)

// Linux entry point
VulkanExample *vulkanExample;
#if defined(VK_USE_PLATFORM_XCB_KHR)
static void handleEvent(const xcb_generic_event_t *event)
{
	if (vulkanExample != NULL)
	{
		vulkanExample->handleEvent(event);
	}
}
#else
static void handleEvent()
{
}
#endif
int main(const int argc, const char *argv[])
{
	for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
	vulkanExample = new VulkanExample();
	vulkanExample->initVulkan();
	vulkanExample->setupWindow();
	vulkanExample->prepare();
	vulkanExample->renderLoop();
	delete(vulkanExample);
	return 0;
}
#elif (defined(VK_USE_PLATFORM_MACOS_MVK) && defined(VK_EXAMPLE_XCODE_GENERATED))
VulkanExample *vulkanExample;
int main(const int argc, const char *argv[])
{
	@autoreleasepool
	{
		for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
		vulkanExample = new VulkanExample();
		vulkanExample->initVulkan();
		vulkanExample->setupWindow(nullptr);
		vulkanExample->prepare();
		vulkanExample->renderLoop();
		delete(vulkanExample);
	}
	return 0;
}
#endif
